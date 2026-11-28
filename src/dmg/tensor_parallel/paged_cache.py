"""
Paged Attention and virtual memory block manager for Distributed KV-Cache.
Eliminates memory fragmentation during multi-turn LLM inference by allocating
fixed-size physical memory blocks per rank (PagedAttention pattern).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from dmg.tensor_parallel.exceptions import TensorParallelError


@dataclass
class PhysicalBlock:
    """A fixed-capacity contiguous physical memory block holding key and value tokens."""

    block_id: int
    block_size: int  # Number of token slots
    head_dim: int
    num_heads: int
    num_tokens_stored: int = 0
    # Shape: [block_size, num_heads * head_dim]
    keys: List[List[float]] = field(default_factory=list)
    values: List[List[float]] = field(default_factory=list)

    @property
    def is_full(self) -> bool:
        return self.num_tokens_stored >= self.block_size

    @property
    def free_slots(self) -> int:
        return self.block_size - self.num_tokens_stored

    def append_token(self, k_vec: List[float], v_vec: List[float]) -> bool:
        """Append a single token key/value pair into this block."""
        if self.is_full:
            return False
        self.keys.append(copy.deepcopy(k_vec))
        self.values.append(copy.deepcopy(v_vec))
        self.num_tokens_stored += 1
        return True

    def clear(self) -> None:
        """Reset the physical block for recycling."""
        self.keys.clear()
        self.values.clear()
        self.num_tokens_stored = 0


@dataclass
class BlockTable:
    """Maintains logical-to-physical block mapping for an active inference request."""

    request_id: str
    physical_block_ids: List[int] = field(default_factory=list)
    total_tokens: int = 0


class PagedKVCacheManager:
    """
    Manages physical memory blocks and per-request block tables for a worker rank.
    Eliminates internal and external memory fragmentation in the distributed KV cache.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Preallocate physical block storage
        self.blocks: List[PhysicalBlock] = [
            PhysicalBlock(
                block_id=i,
                block_size=block_size,
                head_dim=head_dim,
                num_heads=num_heads,
            )
            for i in range(num_blocks)
        ]

        # Free block index pool
        self.free_block_ids: List[int] = list(range(num_blocks))

        # Map: (request_id, layer_idx) -> BlockTable
        self.block_tables: Dict[Tuple[str, int], BlockTable] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    @property
    def memory_utilization_ratio(self) -> float:
        """Ratio of allocated blocks to total pool capacity."""
        allocated = self.num_blocks - self.num_free_blocks
        return allocated / self.num_blocks if self.num_blocks > 0 else 0.0

    def _allocate_block(self) -> int:
        """Pop a free block ID from the pool."""
        if not self.free_block_ids:
            raise TensorParallelError(
                f"Out of physical KV-cache blocks: all {self.num_blocks} blocks are in use."
            )
        return self.free_block_ids.pop()

    def get_or_create_table(self, request_id: str, layer_idx: int) -> BlockTable:
        """Retrieve existing block table or create an empty one."""
        key = (request_id, layer_idx)
        if key not in self.block_tables:
            self.block_tables[key] = BlockTable(request_id=request_id)
        return self.block_tables[key]

    def append_tokens(
        self,
        request_id: str,
        layer_idx: int,
        new_k: List[List[float]],
        new_v: List[List[float]],
    ) -> None:
        """
        Append new key and value representations into the paged cache.
        Automatically allocates new physical blocks as needed.
        """
        table = self.get_or_create_table(request_id, layer_idx)
        num_tokens = len(new_k)

        for i in range(num_tokens):
            k_vec = new_k[i]
            v_vec = new_v[i]

            # Need a new physical block if no blocks allocated or current block is full
            if not table.physical_block_ids:
                new_block_id = self._allocate_block()
                table.physical_block_ids.append(new_block_id)

            last_block_id = table.physical_block_ids[-1]
            last_block = self.blocks[last_block_id]

            if last_block.is_full:
                new_block_id = self._allocate_block()
                table.physical_block_ids.append(new_block_id)
                last_block = self.blocks[new_block_id]

            last_block.append_token(k_vec, v_vec)
            table.total_tokens += 1

    def read_contiguous_sequence(
        self,
        request_id: str,
        layer_idx: int,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Reconstruct the full contiguous (Keys, Values) sequence across allocated blocks.
        """
        table = self.get_or_create_table(request_id, layer_idx)
        if not table.physical_block_ids:
            return [], []

        all_k = []
        all_v = []
        for block_id in table.physical_block_ids:
            block = self.blocks[block_id]
            all_k.extend(block.keys)
            all_v.extend(block.values)

        return all_k, all_v

    def free_request(self, request_id: str, layer_idx: Optional[int] = None) -> None:
        """Release allocated physical blocks back to the free pool."""
        keys_to_remove = [
            k for k in self.block_tables.keys()
            if k[0] == request_id and (layer_idx is None or k[1] == layer_idx)
        ]

        for k in keys_to_remove:
            table = self.block_tables.pop(k)
            for block_id in table.physical_block_ids:
                self.blocks[block_id].clear()
                self.free_block_ids.append(block_id)
