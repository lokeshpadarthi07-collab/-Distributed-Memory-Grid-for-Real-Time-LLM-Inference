"""
Abstract contracts and protocols for Tensor Parallelism and Attention modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from dmg.core.interfaces import AttentionParallelismProvider


@runtime_checkable
class CommunicatorProtocol(Protocol):
    """Protocol for cross-rank collective communication."""

    @property
    def rank(self) -> int:
        """Local worker rank index."""
        ...

    @property
    def world_size(self) -> int:
        """Total number of participating ranks."""
        ...

    def all_reduce_sum(self, tensor: Any) -> Any:
        """Perform All-Reduce Sum across all ranks in the group."""
        ...

    def all_gather(self, tensor: Any, dim: int = 0) -> Any:
        """Perform All-Gather concatenating local slices from each rank."""
        ...

    def broadcast(self, tensor: Any, src_rank: int = 0) -> Any:
        """Broadcast tensor from src_rank to all ranks."""
        ...


@runtime_checkable
class TensorSplitterProtocol(Protocol):
    """Protocol for splitting tensors across tensor parallel ranks."""

    def split_column(self, tensor: Any, world_size: int, rank: int) -> Any:
        """Split 2D weight matrix along columns (output dimension)."""
        ...

    def split_row(self, tensor: Any, world_size: int, rank: int) -> Any:
        """Split 2D weight matrix along rows (input dimension)."""
        ...

    def split_heads(
        self,
        weight: Any,
        num_heads: int,
        world_size: int,
        rank: int,
        dim: int = 0,
    ) -> Any:
        """Split attention weight matrix by head chunks."""
        ...


@runtime_checkable
class ParallelAttentionProtocol(Protocol):
    """Protocol for a rank-local parallel self-attention block."""

    def forward(
        self,
        hidden_states: Any,
        attention_mask: Optional[Any] = None,
        kv_cache: Optional[Any] = None,
    ) -> Tuple[Any, Optional[Any]]:
        """Compute localized attention forward pass and return (output, updated_kv_cache)."""
        ...
