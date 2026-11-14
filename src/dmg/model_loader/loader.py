"""
Optimized LLM Model Weight Loader (Week 1 & Week 2).

Coordinates `LLMModelConfig`, `MemoryMappedWeightStore`, `WeightDistributor`,
and `TensorIntegrityValidator` with thread-pool background prefetching,
bounded LRU tensor caching, on-the-fly dtype conversion, and cluster shard
preparation.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

import numpy as np

from dmg.core.exceptions import ConfigurationError
from dmg.core.interfaces import (
    AttentionParallelismProvider,
    FFNParallelismProvider,
    TensorMetadata,
    WeightLoaderProtocol,
    WorkerShardTarget,
)
from dmg.model_loader.config import ConfigValidator, LLMModelConfig, normalize_dtype
from dmg.model_loader.mmap_loader import MemoryMappedWeightStore
from dmg.model_loader.safetensors_io import SAFETENSORS_TO_NP_DTYPE
from dmg.model_loader.sharding import NodeShardPackage, WeightDistributor
from dmg.model_loader.validator import (
    CheckpointValidationReport,
    TensorIntegrityValidator,
)


@dataclass
class TransformerLayerWeights:
    """Container holding all parameter arrays for one Transformer layer."""

    layer_idx: int
    input_layernorm: np.ndarray[Any, Any]
    post_attention_layernorm: np.ndarray[Any, Any]
    q_proj: np.ndarray[Any, Any]
    k_proj: np.ndarray[Any, Any]
    v_proj: np.ndarray[Any, Any]
    o_proj: np.ndarray[Any, Any]
    gate_proj: np.ndarray[Any, Any]
    up_proj: np.ndarray[Any, Any]
    down_proj: np.ndarray[Any, Any]


class LLMWeightLoader(
    WeightLoaderProtocol,
    AttentionParallelismProvider,
    FFNParallelismProvider,
):
    """
    Enterprise LLM weight loader backed by zero-copy memory-mapped Safetensors.

    Week 2 Optimizations:
    - Bounded thread-safe LRU tensor cache to avoid repeated dtype casts or page faults.
    - Background `ThreadPoolExecutor` layer prefetching (`prefetch_layers`) to overlap
      disk/mmap page warming with compute or network transfers.
    - Optional on-the-fly dtype conversion (`target_dtype`, e.g., F32 -> F16).
    - Direct integration with `WeightDistributor` and `TensorIntegrityValidator`.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        config: Optional[LLMModelConfig] = None,
        *,
        cache_capacity_tensors: int = 64,
        prefetch_workers: int = 2,
        target_dtype: Optional[str] = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        if config is not None:
            self.config = config
            ConfigValidator.validate(self.config)
        else:
            self.config = LLMModelConfig.from_json(self.checkpoint_dir)

        self.target_dtype: Optional[str] = (
            normalize_dtype(target_dtype) if target_dtype is not None else None
        )
        self.cache_capacity = max(0, cache_capacity_tensors)
        self._cache: OrderedDict[str, np.ndarray[Any, Any]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._lock = RLock()

        self.store = MemoryMappedWeightStore(self.checkpoint_dir)
        self.distributor = WeightDistributor(self.config, self.store)
        self.validator = TensorIntegrityValidator(self.config)

        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=prefetch_workers, thread_name_prefix="dmg-prefetch")
            if prefetch_workers > 0
            else None
        )
        self._prefetch_futures: Dict[int, Future[TransformerLayerWeights]] = {}

    def get_tensor_metadata(self, tensor_name: str) -> TensorMetadata:
        """Return metadata for a specific model parameter tensor."""
        return self.store.get_tensor_metadata(tensor_name)

    def load_tensor(
        self,
        tensor_name: str,
        *,
        copy: bool = False,
    ) -> np.ndarray[Any, Any]:
        """
        Load a parameter tensor by name via memory mapping, applying optional
        dtype casting and bounded LRU caching.
        """
        with self._lock:
            if tensor_name in self._cache:
                self._cache.move_to_end(tensor_name)
                self._cache_hits += 1
                cached = self._cache[tensor_name]
                return cached.copy() if copy else cached
            self._cache_misses += 1

        arr = self.store.get_tensor(tensor_name, copy=copy)
        if self.target_dtype is not None:
            target_np_dtype = SAFETENSORS_TO_NP_DTYPE[self.target_dtype]
            if arr.dtype != target_np_dtype:
                arr = arr.astype(target_np_dtype)

        if self.cache_capacity > 0:
            with self._lock:
                self._cache[tensor_name] = arr
                self._cache.move_to_end(tensor_name)
                while len(self._cache) > self.cache_capacity:
                    self._cache.popitem(last=False)

        return arr

    def list_tensors(self) -> List[str]:
        """List all tensor names in the checkpoint."""
        return self.store.list_tensors()

    def load_embeddings(self, *, copy: bool = False) -> Dict[str, np.ndarray[Any, Any]]:
        """Load input token embeddings, final RMSNorm, and output LM head weights."""
        result: Dict[str, np.ndarray[Any, Any]] = {
            "embed_tokens": self.load_tensor("model.embed_tokens.weight", copy=copy),
            "norm": self.load_tensor("model.norm.weight", copy=copy),
        }
        if self.config.tie_word_embeddings:
            result["lm_head"] = result["embed_tokens"]
        else:
            result["lm_head"] = self.load_tensor("lm_head.weight", copy=copy)
        return result

    def load_layer(
        self,
        layer_idx: int,
        *,
        copy: bool = False,
    ) -> TransformerLayerWeights:
        """
        Load all attention, MLP, and normalization weights for `layer_idx`.
        If the layer was scheduled via `prefetch_layers`, awaits the prefetched future.
        """
        if layer_idx < 0 or layer_idx >= self.config.num_hidden_layers:
            raise ConfigurationError(
                f"layer_idx {layer_idx} out of range [0, {self.config.num_hidden_layers})"
            )

        future: Optional[Future[TransformerLayerWeights]] = None
        with self._lock:
            future = self._prefetch_futures.pop(layer_idx, None)
        if future is not None and not copy:
            return future.result()

        return self._load_layer_sync(layer_idx, copy=copy)

    def _load_layer_sync(
        self,
        layer_idx: int,
        *,
        copy: bool = False,
    ) -> TransformerLayerWeights:
        prefix = f"model.layers.{layer_idx}"
        return TransformerLayerWeights(
            layer_idx=layer_idx,
            input_layernorm=self.load_tensor(f"{prefix}.input_layernorm.weight", copy=copy),
            post_attention_layernorm=self.load_tensor(
                f"{prefix}.post_attention_layernorm.weight", copy=copy
            ),
            q_proj=self.load_tensor(f"{prefix}.self_attn.q_proj.weight", copy=copy),
            k_proj=self.load_tensor(f"{prefix}.self_attn.k_proj.weight", copy=copy),
            v_proj=self.load_tensor(f"{prefix}.self_attn.v_proj.weight", copy=copy),
            o_proj=self.load_tensor(f"{prefix}.self_attn.o_proj.weight", copy=copy),
            gate_proj=self.load_tensor(f"{prefix}.mlp.gate_proj.weight", copy=copy),
            up_proj=self.load_tensor(f"{prefix}.mlp.up_proj.weight", copy=copy),
            down_proj=self.load_tensor(f"{prefix}.mlp.down_proj.weight", copy=copy),
        )

    def prefetch_layers(self, layer_indices: Iterable[int]) -> None:
        """
        Asynchronously prefetch one or more Transformer layers into the LRU cache
        using the background worker thread pool.
        """
        if self._executor is None:
            return
        with self._lock:
            for idx in layer_indices:
                if 0 <= idx < self.config.num_hidden_layers and idx not in self._prefetch_futures:
                    self._prefetch_futures[idx] = self._executor.submit(
                        self._load_layer_sync, idx, copy=False
                    )

    def iter_layers(
        self,
        *,
        copy: bool = False,
        prefetch_ahead: int = 1,
    ) -> Iterator[TransformerLayerWeights]:
        """
        Yield `TransformerLayerWeights` sequentially while automatically prefetching
        `prefetch_ahead` upcoming layers in the background.
        """
        total = self.config.num_hidden_layers
        for idx in range(total):
            if prefetch_ahead > 0:
                upcoming = [
                    next_idx
                    for next_idx in range(idx + 1, min(total, idx + 1 + prefetch_ahead))
                ]
                self.prefetch_layers(upcoming)
            yield self.load_layer(idx, copy=copy)

    def get_attention_shard_spec(
        self, layer_idx: int, rank: int, world_size: int
    ) -> Dict[str, Any]:
        """Delegate attention head sharding to `WeightDistributor` (Member 3)."""
        return self.distributor.get_attention_shard_spec(layer_idx, rank, world_size)

    def get_ffn_shard_spec(
        self, layer_idx: int, rank: int, world_size: int
    ) -> Dict[str, Any]:
        """Delegate feed-forward sharding to `WeightDistributor` (Member 4)."""
        return self.distributor.get_ffn_shard_spec(layer_idx, rank, world_size)

    def validate_checkpoint(
        self,
        *,
        strict_coverage: bool = True,
        check_nan_inf: bool = True,
        disallow_all_zeros: bool = False,
    ) -> CheckpointValidationReport:
        """Validate all checkpoint tensor shapes, dtypes, and numerical finiteness."""
        return self.validator.validate_store(
            self.store,
            strict_coverage=strict_coverage,
            check_nan_inf=check_nan_inf,
            disallow_all_zeros=disallow_all_zeros,
        )

    def prepare_distributed_shards(
        self,
        workers: List[WorkerShardTarget],
        *,
        validate: bool = True,
    ) -> Dict[int, NodeShardPackage]:
        """
        Partition model weights across `workers` (Member 1) and optionally verify
        SHA-256 and reconstruction integrity across all ranks.
        """
        packages = self.distributor.prepare_cluster_distribution(
            workers, compute_checksums=True
        )
        if validate:
            self.validator.validate_distributed_packages(packages, store=self.store)
        return packages

    def cache_stats(self) -> Dict[str, int]:
        """Return telemetry on the internal LRU weight cache."""
        with self._lock:
            return {
                "cached_tensors": len(self._cache),
                "cache_capacity": self.cache_capacity,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
            }

    def close(self) -> None:
        """Shut down background prefetch threads, clear cache, and close mmap handles."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        with self._lock:
            self._prefetch_futures.clear()
            self._cache.clear()
        self.store.close()

    def __enter__(self) -> "LLMWeightLoader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
