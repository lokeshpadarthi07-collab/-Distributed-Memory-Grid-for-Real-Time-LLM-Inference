"""
Configuration classes for Tensor Parallelism and Distributed Multi-Head Attention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from dmg.tensor_parallel.exceptions import InvalidParallelConfigError


class AttentionType(str, Enum):
    """Types of multi-head attention supported by the sharding grid."""

    MHA = "multi_head_attention"
    GQA = "grouped_query_attention"
    MQA = "multi_query_attention"


class CommBackend(str, Enum):
    """Supported distributed communication backends."""

    NCCL = "nccl"
    GLOO = "gloo"
    MOCK = "mock"


@dataclass(frozen=True)
class TensorParallelConfig:
    """Core configuration for Tensor Parallelism across worker ranks."""

    world_size: int = 1
    rank: int = 0
    backend: CommBackend = CommBackend.MOCK
    init_method: str = "tcp://127.0.0.1:29500"
    enable_profiling: bool = False
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise InvalidParallelConfigError(
                f"world_size must be >= 1, got {self.world_size}"
            )
        if not (0 <= self.rank < self.world_size):
            raise InvalidParallelConfigError(
                f"rank must be in range [0, {self.world_size - 1}], got {self.rank}"
            )


@dataclass(frozen=True)
class ParallelAttentionConfig:
    """Model-level hyperparameter configuration for Parallel Self-Attention."""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    bias: bool = False
    dropout_prob: float = 0.0
    attention_type: AttentionType = AttentionType.MHA
    use_rope: bool = False
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096
    use_chunked_attention: bool = False
    chunk_size: int = 64
    sliding_window: Optional[int] = None
    paged_cache_block_size: int = 16
    max_num_paged_blocks: int = 256
    tp_config: TensorParallelConfig = field(default_factory=TensorParallelConfig)

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise InvalidParallelConfigError(
                f"hidden_size must be > 0, got {self.hidden_size}"
            )
        if self.num_attention_heads <= 0:
            raise InvalidParallelConfigError(
                f"num_attention_heads must be > 0, got {self.num_attention_heads}"
            )

        # Default num_key_value_heads to num_attention_heads (MHA)
        if self.num_key_value_heads is None:
            object.__setattr__(self, "num_key_value_heads", self.num_attention_heads)

        # Calculate head dimension if not explicitly set
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise InvalidParallelConfigError(
                    f"hidden_size ({self.hidden_size}) must be divisible by "
                    f"num_attention_heads ({self.num_attention_heads})"
                )
            object.__setattr__(
                self, "head_dim", self.hidden_size // self.num_attention_heads
            )

        # Validate head divisibility for Tensor Parallelism
        ws = self.tp_config.world_size
        if self.num_attention_heads % ws != 0:
            raise InvalidParallelConfigError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"world_size ({ws})"
            )

        if self.num_key_value_heads is not None and self.num_key_value_heads < ws:
            if ws % self.num_key_value_heads != 0:
                raise InvalidParallelConfigError(
                    f"world_size ({ws}) must be a multiple of num_key_value_heads "
                    f"({self.num_key_value_heads}) for GQA/MQA replication"
                )

    @property
    def local_num_heads(self) -> int:
        """Number of query attention heads handled by the local rank."""
        return self.num_attention_heads // self.tp_config.world_size

    @property
    def local_num_kv_heads(self) -> int:
        """Number of key-value attention heads handled by the local rank."""
        num_kv = self.num_key_value_heads or self.num_attention_heads
        ws = self.tp_config.world_size
        if num_kv >= ws:
            return num_kv // ws
        return 1  # Replicated KV head across ranks

    @property
    def local_hidden_size(self) -> int:
        """Total dimension of query heads handled by local rank."""
        assert self.head_dim is not None
        return self.local_num_heads * self.head_dim
