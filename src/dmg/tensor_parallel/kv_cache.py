"""
Distributed KV-Cache management module for Tensor Parallelism.
Manages rank-local Key-Value cache buffers during multi-head autoregressive decoding.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dmg.tensor_parallel.config import ParallelAttentionConfig
from dmg.tensor_parallel.exceptions import TensorParallelError


@dataclass
class RankKVCacheState:
    """Stores the key and value cache for a specific rank and layer."""

    layer_idx: int
    rank: int
    max_seq_len: int
    current_seq_len: int = 0
    key_cache: Optional[List[List[float]]] = None    # [current_seq_len, num_local_kv_heads * head_dim]
    value_cache: Optional[List[List[float]]] = None  # [current_seq_len, num_local_kv_heads * head_dim]

    @property
    def is_empty(self) -> bool:
        return self.current_seq_len == 0

    @property
    def remaining_capacity(self) -> int:
        return self.max_seq_len - self.current_seq_len


class DistributedKVCacheManager:
    """
    Manages KV-cache lifecycle across layers and tensor parallel ranks.
    Supports continuous sequence expansion during autoregressive generation.
    """

    def __init__(self, config: ParallelAttentionConfig, max_seq_len: int = 4096):
        self.config = config
        self.max_seq_len = max_seq_len
        # Map: layer_idx -> RankKVCacheState
        self._layers: Dict[int, RankKVCacheState] = {}

    def get_or_create(self, layer_idx: int) -> RankKVCacheState:
        """Retrieve existing cache for layer or initialize a new empty state."""
        if layer_idx not in self._layers:
            self._layers[layer_idx] = RankKVCacheState(
                layer_idx=layer_idx,
                rank=self.config.tp_config.rank,
                max_seq_len=self.max_seq_len,
            )
        return self._layers[layer_idx]

    def update_cache(
        self,
        layer_idx: int,
        new_k: List[List[float]],
        new_v: List[List[float]],
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Append new key and value representations to the local rank cache.
        Returns the complete concatenated past + present (K, V) representations.
        """
        state = self.get_or_create(layer_idx)
        num_new_tokens = len(new_k)

        if state.current_seq_len + num_new_tokens > state.max_seq_len:
            raise TensorParallelError(
                f"KV Cache overflow: capacity {state.max_seq_len} exceeded by "
                f"{state.current_seq_len + num_new_tokens} tokens"
            )

        if state.is_empty:
            state.key_cache = copy.deepcopy(new_k)
            state.value_cache = copy.deepcopy(new_v)
        else:
            assert state.key_cache is not None and state.value_cache is not None
            state.key_cache.extend(copy.deepcopy(new_k))
            state.value_cache.extend(copy.deepcopy(new_v))

        state.current_seq_len += num_new_tokens
        assert state.key_cache is not None and state.value_cache is not None
        return state.key_cache, state.value_cache

    def clear(self, layer_idx: Optional[int] = None) -> None:
        """Reset cache for a specific layer or all layers."""
        if layer_idx is not None:
            self._layers.pop(layer_idx, None)
        else:
            self._layers.clear()

    def get_memory_bytes(self) -> int:
        """Estimate total RAM bytes occupied by cached floats across all layers."""
        total_elements = 0
        for state in self._layers.values():
            if state.key_cache:
                total_elements += len(state.key_cache) * len(state.key_cache[0])
            if state.value_cache:
                total_elements += len(state.value_cache) * len(state.value_cache[0])
        return total_elements * 4  # 4 bytes per float32
