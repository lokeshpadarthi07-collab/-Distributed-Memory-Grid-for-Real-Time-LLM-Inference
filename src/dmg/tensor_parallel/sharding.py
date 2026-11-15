"""
Attention head sharding and weight distribution engine.
Partitions Query, Key, and Value weights across distributed worker nodes
for MHA (Multi-Head Attention), GQA (Grouped-Query Attention), and MQA (Multi-Query Attention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dmg.core.interfaces import AttentionParallelismProvider, WorkerShardTarget
from dmg.tensor_parallel.config import (
    AttentionType,
    ParallelAttentionConfig,
    TensorParallelConfig,
)
from dmg.tensor_parallel.exceptions import AttentionHeadSplitError
from dmg.tensor_parallel.splitting import TensorSplitter


@dataclass(frozen=True)
class HeadPartition:
    """Metadata describing the attention head assignment for a specific rank."""

    rank: int
    world_size: int
    query_heads_range: Tuple[int, int]  # [start_head, end_head)
    kv_heads_range: Tuple[int, int]     # [start_head, end_head)
    is_kv_replicated: bool = False

    @property
    def num_query_heads(self) -> int:
        return self.query_heads_range[1] - self.query_heads_range[0]

    @property
    def num_kv_heads(self) -> int:
        return self.kv_heads_range[1] - self.kv_heads_range[0]


@dataclass
class AttentionShardManifest:
    """Complete sharded weight and partition manifest for a rank."""

    layer_idx: int
    rank: int
    world_size: int
    partition: HeadPartition
    q_weight_shard: Any
    k_weight_shard: Any
    v_weight_shard: Any
    o_weight_shard: Any
    q_bias_shard: Optional[Any] = None
    k_bias_shard: Optional[Any] = None
    v_bias_shard: Optional[Any] = None
    o_bias: Optional[Any] = None


class AttentionHeadSharder(AttentionParallelismProvider):
    """
    Production head sharder that implements the AttentionParallelismProvider protocol.
    Slices monolithic model weights into localized rank slices.
    """

    def __init__(self, config: ParallelAttentionConfig):
        self.config = config

    def compute_head_partition(self, rank: int, world_size: int) -> HeadPartition:
        """Calculate start and end head indices for Query and KV heads for a rank."""
        if rank < 0 or rank >= world_size:
            raise AttentionHeadSplitError(
                f"Rank {rank} out of range [0, {world_size - 1}]"
            )

        num_q_heads = self.config.num_attention_heads
        if num_q_heads % world_size != 0:
            raise AttentionHeadSplitError(
                f"num_attention_heads ({num_q_heads}) not divisible by world_size ({world_size})"
            )

        q_per_rank = num_q_heads // world_size
        q_start = rank * q_per_rank
        q_end = q_start + q_per_rank

        num_kv_heads = self.config.num_key_value_heads or num_q_heads
        if num_kv_heads >= world_size:
            if num_kv_heads % world_size != 0:
                raise AttentionHeadSplitError(
                    f"num_key_value_heads ({num_kv_heads}) not divisible by world_size ({world_size})"
                )
            kv_per_rank = num_kv_heads // world_size
            kv_start = rank * kv_per_rank
            kv_end = kv_start + kv_per_rank
            is_replicated = False
        else:
            # When KV heads < world_size (e.g. MQA with 1 head across 4 ranks)
            # Replicate KV heads across ranks in the group
            kv_start = 0
            kv_end = num_kv_heads
            is_replicated = True

        return HeadPartition(
            rank=rank,
            world_size=world_size,
            query_heads_range=(q_start, q_end),
            kv_heads_range=(kv_start, kv_end),
            is_kv_replicated=is_replicated,
        )

    def shard_layer_weights(
        self,
        layer_idx: int,
        rank: int,
        world_size: int,
        q_weight: Any,
        k_weight: Any,
        v_weight: Any,
        o_weight: Any,
        q_bias: Optional[Any] = None,
        k_bias: Optional[Any] = None,
        v_bias: Optional[Any] = None,
        o_bias: Optional[Any] = None,
    ) -> AttentionShardManifest:
        """
        Slice full attention layer weights into rank-local slices.
        - W_q, W_k, W_v are column-parallel (split on output dim 0).
        - W_o is row-parallel (split on input dim 1).
        """
        partition = self.compute_head_partition(rank, world_size)
        head_dim = self.config.head_dim
        assert head_dim is not None

        # Shard W_q
        q_shard = TensorSplitter.split_attention_heads(
            q_weight,
            num_heads=self.config.num_attention_heads,
            head_dim=head_dim,
            world_size=world_size,
            rank=rank,
            dim=0,
        )

        # Shard W_k
        if partition.is_kv_replicated:
            k_shard = k_weight  # Replicated full KV head
            v_shard = v_weight
        else:
            num_kv = self.config.num_key_value_heads or self.config.num_attention_heads
            k_shard = TensorSplitter.split_attention_heads(
                k_weight,
                num_heads=num_kv,
                head_dim=head_dim,
                world_size=world_size,
                rank=rank,
                dim=0,
            )
            v_shard = TensorSplitter.split_attention_heads(
                v_weight,
                num_heads=num_kv,
                head_dim=head_dim,
                world_size=world_size,
                rank=rank,
                dim=0,
            )

        # Shard W_o: row-parallel (dim 1)
        o_shard = TensorSplitter.split_row(
            o_weight, world_size=world_size, rank=rank, dim=1
        )

        # Biases
        q_b_shard = (
            TensorSplitter.split_along_dim(q_bias, dim=0, world_size=world_size, rank=rank)
            if q_bias is not None
            else None
        )
        k_b_shard = (
            (k_bias if partition.is_kv_replicated else TensorSplitter.split_along_dim(k_bias, dim=0, world_size=world_size, rank=rank))
            if k_bias is not None
            else None
        )
        v_b_shard = (
            (v_bias if partition.is_kv_replicated else TensorSplitter.split_along_dim(v_bias, dim=0, world_size=world_size, rank=rank))
            if v_bias is not None
            else None
        )

        return AttentionShardManifest(
            layer_idx=layer_idx,
            rank=rank,
            world_size=world_size,
            partition=partition,
            q_weight_shard=q_shard,
            k_weight_shard=k_shard,
            v_weight_shard=v_shard,
            o_weight_shard=o_shard,
            q_bias_shard=q_b_shard,
            k_bias_shard=k_b_shard,
            v_bias_shard=v_b_shard,
            o_bias=o_bias,  # o_bias is added after all-reduce synchronization
        )

    def get_attention_shard_spec(
        self, layer_idx: int, rank: int, world_size: int
    ) -> Dict[str, Any]:
        """
        Implementation of cross-module AttentionParallelismProvider protocol.
        Returns a dictionary specification of head ranges and slice dimensions.
        """
        partition = self.compute_head_partition(rank, world_size)
        return {
            "layer_idx": layer_idx,
            "rank": rank,
            "world_size": world_size,
            "query_heads_start": partition.query_heads_range[0],
            "query_heads_end": partition.query_heads_range[1],
            "num_query_heads": partition.num_query_heads,
            "kv_heads_start": partition.kv_heads_range[0],
            "kv_heads_end": partition.kv_heads_range[1],
            "num_kv_heads": partition.num_kv_heads,
            "is_kv_replicated": partition.is_kv_replicated,
            "head_dim": self.config.head_dim,
            "local_hidden_size": self.config.local_hidden_size,
        }
