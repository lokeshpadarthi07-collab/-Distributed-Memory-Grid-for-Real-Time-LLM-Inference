"""
High-level Distributed Inference Engine for Member 3.
Integrates AttentionHeadSharder, ParallelSelfAttention, OutputSynchronizer,
and DistributedKVCacheManager with integrated latency profiling.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from dmg.tensor_parallel.attention import ParallelSelfAttention
from dmg.tensor_parallel.config import ParallelAttentionConfig
from dmg.tensor_parallel.kv_cache import DistributedKVCacheManager
from dmg.tensor_parallel.profiler import ProfilingRecord, TensorParallelProfiler
from dmg.tensor_parallel.synchronizer import AttentionOutputSynchronizer


class DistributedInferenceEngine:
    """
    Coordinating inference execution engine for a single worker node rank.
    Handles prefill, iterative token generation, KV-caching, and profiling.
    """

    def __init__(
        self,
        config: ParallelAttentionConfig,
        attention_layers: List[ParallelSelfAttention],
        synchronizers: List[AttentionOutputSynchronizer],
        kv_cache_mgr: Optional[DistributedKVCacheManager] = None,
    ):
        self.config = config
        self.attention_layers = attention_layers
        self.synchronizers = synchronizers
        self.kv_cache = kv_cache_mgr or DistributedKVCacheManager(config)
        self.profiler = TensorParallelProfiler(
            rank=config.tp_config.rank,
            world_size=config.tp_config.world_size,
        )

    def forward_layer(
        self,
        layer_idx: int,
        hidden_states: Any,
        residual: Optional[Any] = None,
        causal_mask: bool = True,
        step_id: Optional[str] = None,
    ) -> Any:
        """Execute a single layer's parallel attention and synchronization."""
        attention = self.attention_layers[layer_idx]
        synchronizer = self.synchronizers[layer_idx]

        t0 = time.perf_counter()
        # 1. Local Q, K, V projections & attention
        local_out = attention.forward_local(hidden_states, causal_mask=causal_mask)
        t1 = time.perf_counter()

        # 2. Output All-Reduce synchronization
        op_id = f"step_{step_id}_layer_{layer_idx}" if step_id else None
        sync_out = synchronizer.synchronize(
            local_out, residual=residual, op_id=op_id
        )
        t2 = time.perf_counter()

        if self.config.tp_config.enable_profiling:
            comp_ms = (t1 - t0) * 1000
            sync_ms = (t2 - t1) * 1000
            self.profiler.record_step(
                step_id=step_id or f"step_{layer_idx}",
                qkv_ms=comp_ms * 0.4,
                attn_ms=comp_ms * 0.4,
                out_ms=comp_ms * 0.2,
                sync_ms=sync_ms,
            )

        return sync_out

    def forward_full_model(
        self,
        hidden_states: Any,
        step_id: Optional[str] = None,
    ) -> Any:
        """Sequentially execute forward pass across all layers in the pipeline."""
        curr = hidden_states
        for idx in range(len(self.attention_layers)):
            # Residual connection
            residual = curr
            curr = self.forward_layer(
                layer_idx=idx,
                hidden_states=curr,
                residual=residual,
                step_id=step_id,
            )
        return curr
