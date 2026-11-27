"""
Adaptive Dynamic Layer Eviction and High-Watermark Memory Management Engine
for the Distributed Memory Grid.

Monitors process virtual memory pressure and coordinates with Member 5 (KV-Cache)
and Member 1 (Master-Worker Network) to dynamically evict non-active layer weights
or prioritize critical transformer projections when VRAM/RAM budgets are saturated.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from dmg.core.interfaces import TensorMetadata


class EvictionPolicy(str, Enum):
    """Eviction heuristics for layer and tensor caches under memory pressure."""

    LRU = "lru"
    FIFO = "fifo"
    PRIORITY_AWARE = "priority_aware"


@dataclass(frozen=True)
class TensorEvictionCandidate:
    """Descriptor for a cached weight tensor eligible for eviction."""

    tensor_name: str
    size_bytes: int
    last_accessed: float
    priority_score: int  # Higher score = more critical (less likely to evict)


@dataclass
class EvictionCycleResult:
    """Report detailing reclaimed memory from an eviction pass."""

    freed_bytes: int
    evicted_tensors: List[str]
    remaining_bytes: int
    target_budget_bytes: int
    policy_used: EvictionPolicy


class DynamicLayerEvictionManager:
    """
    Monitors working memory set of loaded transformer parameters and proactively
    enforces maximum memory budgets by evicting low-priority or least-recently used layers.
    """

    # Default priorities: Embeddings & norms are highly reused, MLP down_proj/gate_proj can be evicted
    CRITICAL_TENSOR_SUBSTRINGS = ("embed_tokens", "norm", "lm_head")

    def __init__(
        self,
        max_budget_bytes: int,
        *,
        high_watermark_ratio: float = 0.85,
        low_watermark_ratio: float = 0.70,
        policy: EvictionPolicy = EvictionPolicy.PRIORITY_AWARE,
    ) -> None:
        self.max_budget_bytes = max_budget_bytes
        self.high_watermark_bytes = int(max_budget_bytes * high_watermark_ratio)
        self.low_watermark_bytes = int(max_budget_bytes * low_watermark_ratio)
        self.policy = policy

        self._active_registry: Dict[str, Tuple[int, float]] = {}  # name -> (bytes, timestamp)
        self._pinned_tensors: Set[str] = set()

    @property
    def current_allocated_bytes(self) -> int:
        return sum(sz for sz, _ in self._active_registry.values())

    def pin_tensor(self, tensor_name: str) -> None:
        """Pin a tensor in memory to prevent eviction (e.g. during active forward pass)."""
        self._pinned_tensors.add(tensor_name)

    def unpin_tensor(self, tensor_name: str) -> None:
        """Unpin a tensor so it can be evicted if memory pressure arises."""
        self._pinned_tensors.discard(tensor_name)

    def register_tensor_access(
        self,
        tensor_name: str,
        size_bytes: int,
        timestamp: float,
    ) -> None:
        """Record or update access metadata for a loaded tensor."""
        self._active_registry[tensor_name] = (size_bytes, timestamp)

    def is_under_pressure(self) -> bool:
        """Check if memory usage has exceeded the high-watermark threshold."""
        return self.current_allocated_bytes >= self.high_watermark_bytes

    def calculate_priority(self, tensor_name: str) -> int:
        """Compute retention priority score for priority-aware eviction."""
        if any(crit in tensor_name for crit in self.CRITICAL_TENSOR_SUBSTRINGS):
            return 100
        if "self_attn" in tensor_name:
            return 50
        return 10  # MLP layers are large and can be repopulated quickly from mmap

    def evict_to_budget(self) -> EvictionCycleResult:
        """
        Reclaim memory by selecting candidate tensors according to `self.policy`
        until memory drops below `self.low_watermark_bytes`.
        """
        if not self.is_under_pressure():
            return EvictionCycleResult(
                freed_bytes=0,
                evicted_tensors=[],
                remaining_bytes=self.current_allocated_bytes,
                target_budget_bytes=self.low_watermark_bytes,
                policy_used=self.policy,
            )

        candidates: List[TensorEvictionCandidate] = []
        for name, (sz, ts) in self._active_registry.items():
            if name in self._pinned_tensors:
                continue
            candidates.append(
                TensorEvictionCandidate(
                    tensor_name=name,
                    size_bytes=sz,
                    last_accessed=ts,
                    priority_score=self.calculate_priority(name),
                )
            )

        if self.policy == EvictionPolicy.LRU:
            candidates.sort(key=lambda c: c.last_accessed)
        elif self.policy == EvictionPolicy.FIFO:
            candidates.sort(key=lambda c: c.last_accessed)
        elif self.policy == EvictionPolicy.PRIORITY_AWARE:
            # Sort by lowest priority first, then oldest access
            candidates.sort(key=lambda c: (c.priority_score, c.last_accessed))

        freed_bytes = 0
        evicted_names: List[str] = []

        for cand in candidates:
            if self.current_allocated_bytes - freed_bytes <= self.low_watermark_bytes:
                break
            del self._active_registry[cand.tensor_name]
            freed_bytes += cand.size_bytes
            evicted_names.append(cand.tensor_name)

        return EvictionCycleResult(
            freed_bytes=freed_bytes,
            evicted_tensors=evicted_names,
            remaining_bytes=self.current_allocated_bytes,
            target_budget_bytes=self.low_watermark_bytes,
            policy_used=self.policy,
        )
