"""
Model weight distribution and tensor-parallel sharding engine (Week 2).

Prepares LLM model weights for distribution across multiple worker nodes:
- Coordinates with Member 1 (Master-Worker Network) by generating serializable
  per-node `NodeShardPackage` manifests and binary Safetensors shard bundles.
- Coordinates with Member 3 (Attention Head Tensor Parallelism) via head-aligned
  column-parallel splitting of `q_proj`, `k_proj`, `v_proj` and row-parallel
  splitting of `o_proj` (supporting both Multi-Head and Grouped-Query Attention).
- Coordinates with Member 4 (Feed-Forward Distribution) via column-parallel
  splitting of `gate_proj`, `up_proj` and row-parallel splitting of `down_proj`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from dmg.core.exceptions import ShardDistributionError
from dmg.core.interfaces import (
    AttentionParallelismProvider,
    FFNParallelismProvider,
    WorkerShardTarget,
)
from dmg.model_loader.config import ConfigValidator, LLMModelConfig
from dmg.model_loader.mmap_loader import MemoryMappedWeightStore
from dmg.model_loader.safetensors_io import save_safetensors_file


class PartitionStrategy(str, Enum):
    """Strategy used to distribute a parameter tensor across worker ranks."""

    REPLICATED = "replicated"
    COLUMN_PARALLEL = "column_parallel"  # Split along output features (dim 0 in [out, in])
    ROW_PARALLEL = "row_parallel"        # Split along input features (dim 1 in [out, in])


@dataclass(frozen=True)
class TensorShardDescriptor:
    """Describes how a single tensor slice is assigned to a specific worker rank."""

    tensor_name: str
    rank: int
    world_size: int
    strategy: PartitionStrategy
    source_shape: Tuple[int, ...]
    shard_shape: Tuple[int, ...]
    split_dim: Optional[int]
    slice_start: int
    slice_end: int
    dtype: str
    sha256: str = ""


@dataclass
class NodeShardPackage:
    """
    Complete weight package prepared for a single worker node (`rank` in `0..world_size-1`).
    Can be transmitted directly over Member 1's gRPC transport or exported to a per-node
    `.safetensors` file.
    """

    target: WorkerShardTarget
    descriptors: Dict[str, TensorShardDescriptor] = field(default_factory=dict)
    tensors: Dict[str, np.ndarray[Any, Any]] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        """Total byte size of all tensor slices in this node package."""
        return sum(arr.nbytes for arr in self.tensors.values())

    def export_safetensors(self, output_dir: Union[str, Path]) -> Path:
        """Write this worker node's shard package to a dedicated `.safetensors` file."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"worker-rank-{self.target.rank:04d}-of-{self.target.world_size:04d}.safetensors"
        )
        file_path = out_dir / filename
        metadata = {
            "node_id": self.target.node_id,
            "rank": str(self.target.rank),
            "world_size": str(self.target.world_size),
        }
        return save_safetensors_file(self.tensors, file_path, metadata=metadata)


class WeightDistributor(AttentionParallelismProvider, FFNParallelismProvider):
    """
    Plans and executes zero-copy memory-mapped tensor partitioning across
    multiple worker nodes for Attention (Member 3) and Feed-Forward (Member 4)
    parallelism.
    """

    def __init__(
        self,
        config: LLMModelConfig,
        store: MemoryMappedWeightStore,
    ) -> None:
        self.config = config
        self.store = store

    @staticmethod
    def classify_tensor_strategy(tensor_name: str) -> Tuple[PartitionStrategy, Optional[int]]:
        """
        Determine the tensor-parallel partitioning strategy and split axis for a
        canonical Transformer weight parameter.

        In Hugging Face PyTorch Linear layers, weights have shape `(out_features, in_features)`:
        - `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj` are Column-Parallel
          (split output dimension = axis 0).
        - `o_proj`, `down_proj` are Row-Parallel (split input dimension = axis 1)
          so their outputs can be all-reduced across nodes without an intermediate gather.
        - Embeddings, RMSNorms, and `lm_head` default to Replicated unless vocabulary
          parallelism is explicitly requested.
        """
        if any(
            proj in tensor_name
            for proj in (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.q_proj.bias",
                "self_attn.k_proj.bias",
                "self_attn.v_proj.bias",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.gate_proj.bias",
                "mlp.up_proj.bias",
            )
        ):
            return PartitionStrategy.COLUMN_PARALLEL, 0

        if any(
            proj in tensor_name
            for proj in (
                "self_attn.o_proj.weight",
                "mlp.down_proj.weight",
            )
        ):
            return PartitionStrategy.ROW_PARALLEL, 1

        return PartitionStrategy.REPLICATED, None

    def extract_tensor_shard(
        self,
        tensor_name: str,
        rank: int,
        world_size: int,
        *,
        compute_checksum: bool = True,
    ) -> Tuple[np.ndarray[Any, Any], TensorShardDescriptor]:
        """
        Slice the requested tensor for worker `rank` out of `world_size` directly
        from the memory-mapped store.
        """
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ShardDistributionError(
                f"Invalid (rank={rank}, world_size={world_size}) when partitioning '{tensor_name}'"
            )

        ConfigValidator.validate_parallelism(self.config, world_size)
        meta = self.store.get_tensor_metadata(tensor_name)
        strategy, split_dim = self.classify_tensor_strategy(tensor_name)

        if strategy == PartitionStrategy.REPLICATED or split_dim is None or world_size == 1:
            shard_arr = self.store.get_tensor(tensor_name, copy=False)
            slice_start = 0
            slice_end = meta.shape[0] if meta.shape else 0
        else:
            dim_size = meta.shape[split_dim]
            if dim_size % world_size != 0:
                raise ShardDistributionError(
                    f"Dimension {split_dim} (size {dim_size}) of tensor '{tensor_name}' "
                    f"is not evenly divisible by world_size={world_size}"
                )
            chunk = dim_size // world_size
            slice_start = rank * chunk
            slice_end = slice_start + chunk
            shard_arr = self.store.get_tensor_slice(
                tensor_name,
                dim=split_dim,
                start=slice_start,
                end=slice_end,
                copy=True,
            )

        checksum = ""
        if compute_checksum:
            checksum = hashlib.sha256(np.ascontiguousarray(shard_arr).tobytes()).hexdigest()

        descriptor = TensorShardDescriptor(
            tensor_name=tensor_name,
            rank=rank,
            world_size=world_size,
            strategy=strategy,
            source_shape=meta.shape,
            shard_shape=tuple(shard_arr.shape),
            split_dim=split_dim,
            slice_start=slice_start,
            slice_end=slice_end,
            dtype=meta.dtype,
            sha256=checksum,
        )
        return shard_arr, descriptor

    def get_attention_shard_spec(
        self, layer_idx: int, rank: int, world_size: int
    ) -> Dict[str, Any]:
        """
        Implementation of `AttentionParallelismProvider` for Member 3.
        Returns head-partitioned `q_proj`, `k_proj`, `v_proj`, and `o_proj` slices
        along with local head counts for `rank`.
        """
        ConfigValidator.validate_parallelism(self.config, world_size)
        if layer_idx < 0 or layer_idx >= self.config.num_hidden_layers:
            raise ShardDistributionError(
                f"layer_idx {layer_idx} out of range [0, {self.config.num_hidden_layers})"
            )

        prefix = f"model.layers.{layer_idx}.self_attn"
        proj_names = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"]
        if self.config.attention_bias:
            proj_names.extend(["q_proj.bias", "k_proj.bias", "v_proj.bias", "o_proj.bias"])

        weights: Dict[str, np.ndarray[Any, Any]] = {}
        descriptors: Dict[str, TensorShardDescriptor] = {}
        for short_name in proj_names:
            full_name = f"{prefix}.{short_name}"
            if full_name in self.store.index.tensor_map:
                arr, desc = self.extract_tensor_shard(full_name, rank, world_size)
                weights[short_name] = arr
                descriptors[short_name] = desc

        assert self.config.num_key_value_heads is not None
        return {
            "layer_idx": layer_idx,
            "rank": rank,
            "world_size": world_size,
            "num_local_attention_heads": self.config.num_attention_heads // world_size,
            "num_local_key_value_heads": self.config.num_key_value_heads // world_size,
            "head_dim": self.config.head_dim,
            "weights": weights,
            "descriptors": descriptors,
        }

    def get_ffn_shard_spec(
        self, layer_idx: int, rank: int, world_size: int
    ) -> Dict[str, Any]:
        """
        Implementation of `FFNParallelismProvider` for Member 4.
        Returns column-parallel `gate_proj` & `up_proj` and row-parallel `down_proj`
        weight slices for `rank`.
        """
        ConfigValidator.validate_parallelism(self.config, world_size)
        if layer_idx < 0 or layer_idx >= self.config.num_hidden_layers:
            raise ShardDistributionError(
                f"layer_idx {layer_idx} out of range [0, {self.config.num_hidden_layers})"
            )

        prefix = f"model.layers.{layer_idx}.mlp"
        proj_names = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
        if self.config.mlp_bias:
            proj_names.extend(["gate_proj.bias", "up_proj.bias", "down_proj.bias"])

        weights: Dict[str, np.ndarray[Any, Any]] = {}
        descriptors: Dict[str, TensorShardDescriptor] = {}
        for short_name in proj_names:
            full_name = f"{prefix}.{short_name}"
            if full_name in self.store.index.tensor_map:
                arr, desc = self.extract_tensor_shard(full_name, rank, world_size)
                weights[short_name] = arr
                descriptors[short_name] = desc

        return {
            "layer_idx": layer_idx,
            "rank": rank,
            "world_size": world_size,
            "local_intermediate_size": self.config.intermediate_size // world_size,
            "hidden_size": self.config.hidden_size,
            "weights": weights,
            "descriptors": descriptors,
        }

    def build_node_package(
        self,
        target: WorkerShardTarget,
        *,
        compute_checksums: bool = True,
    ) -> NodeShardPackage:
        """
        Build a complete `NodeShardPackage` for a specific worker node (`target.rank`).
        """
        ConfigValidator.validate_parallelism(self.config, target.world_size)
        package = NodeShardPackage(target=target)
        for tensor_name in self.store.list_tensors():
            arr, desc = self.extract_tensor_shard(
                tensor_name,
                target.rank,
                target.world_size,
                compute_checksum=compute_checksums,
            )
            package.tensors[tensor_name] = arr
            package.descriptors[tensor_name] = desc
        return package

    def prepare_cluster_distribution(
        self,
        workers: List[WorkerShardTarget],
        *,
        compute_checksums: bool = True,
    ) -> Dict[int, NodeShardPackage]:
        """
        Prepare distributed weight packages for all registered worker nodes in
        Member 1's cluster topology.
        """
        if not workers:
            raise ShardDistributionError("Worker target list cannot be empty.")

        world_size = len(workers)
        ranks = sorted(w.rank for w in workers)
        if ranks != list(range(world_size)) or any(
            w.world_size != world_size for w in workers
        ):
            raise ShardDistributionError(
                f"Worker targets must have contiguous ranks 0..{world_size - 1} "
                f"and matching world_size={world_size}."
            )

        packages: Dict[int, NodeShardPackage] = {}
        for worker in sorted(workers, key=lambda w: w.rank):
            packages[worker.rank] = self.build_node_package(
                worker, compute_checksums=compute_checksums
            )
        return packages

    @staticmethod
    def reconstruct_sharded_tensor(
        shards: List[Tuple[TensorShardDescriptor, np.ndarray[Any, Any]]]
    ) -> np.ndarray[Any, Any]:
        """
        Reconstruct a full weight tensor from its distributed worker shards,
        verifying lossless round-trip partitioning.
        """
        if not shards:
            raise ShardDistributionError("Cannot reconstruct tensor from empty shard list.")

        ordered = sorted(shards, key=lambda item: item[0].rank)
        first_desc, first_arr = ordered[0]
        if first_desc.strategy == PartitionStrategy.REPLICATED or first_desc.split_dim is None:
            return first_arr.copy()

        arrays = [arr for _, arr in ordered]
        return np.concatenate(arrays, axis=first_desc.split_dim)
