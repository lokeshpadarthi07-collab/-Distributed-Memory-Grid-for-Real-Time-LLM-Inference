"""
BF16/FP32 bit-conversion utilities, PyTorch zero-copy tensor bridge, and
cluster memory profiling diagnostics for Member 2's Model Loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from dmg.model_loader.config import LLMModelConfig, dtype_byte_size
from dmg.model_loader.sharding import PartitionStrategy, WeightDistributor

try:
    import torch as _torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    _torch = None  # type: ignore[assignment]
    HAS_TORCH = False


def bf16_words_to_fp32(bf16_uint16: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Convert a uint16 array containing raw bfloat16 (BF16) bit patterns into
    standard IEEE-754 float32 by left-shifting 16 bits into the upper word.
    """
    if bf16_uint16.dtype != np.uint16:
        raise ValueError(
            f"Expected uint16 array of raw BF16 words, got {bf16_uint16.dtype}"
        )
    uint32_bits = bf16_uint16.astype(np.uint32) << 16
    return uint32_bits.view(np.float32)


def fp32_to_bf16_words(fp32_arr: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Convert an IEEE-754 float32 NumPy array into raw bfloat16 (BF16) uint16
    words using round-to-nearest-even bit truncation.
    """
    if fp32_arr.dtype != np.float32:
        fp32_arr = fp32_arr.astype(np.float32)
    uint32_bits = np.ascontiguousarray(fp32_arr).view(np.uint32)
    # Round-to-nearest-even bias
    rounding_bias = 0x00007FFF + ((uint32_bits >> 16) & 1)
    rounded = uint32_bits + rounding_bias
    return (rounded >> 16).astype(np.uint16)


def to_framework_tensor(arr: np.ndarray[Any, Any], *, copy: bool = False) -> Any:
    """
    Bridge a memory-mapped NumPy array to a PyTorch tensor (`torch.from_numpy`)
    when PyTorch is installed in the worker environment (Members 3 & 4),
    falling back transparently to a NumPy ndarray otherwise.
    """
    target = arr.copy() if (copy or not arr.flags.writeable) and HAS_TORCH else arr
    if HAS_TORCH and _torch is not None:
        return _torch.from_numpy(np.ascontiguousarray(target))
    return arr.copy() if copy else arr


@dataclass(frozen=True)
class NodeMemoryProfile:
    """Estimated memory allocation breakdown for a single worker rank."""

    rank: int
    world_size: int
    replicated_bytes: int
    column_parallel_bytes: int
    row_parallel_bytes: int
    total_bytes: int

    @property
    def total_megabytes(self) -> float:
        return self.total_bytes / (1024.0 * 1024.0)


@dataclass
class ClusterMemoryReport:
    """Cluster-wide memory distribution report across `world_size` worker nodes."""

    model_name: str
    world_size: int
    unsharded_model_bytes: int
    per_node_profiles: Dict[int, NodeMemoryProfile] = field(default_factory=dict)

    @property
    def memory_savings_ratio(self) -> float:
        """Ratio of per-node sharded memory to full unsharded model memory."""
        if not self.per_node_profiles or self.unsharded_model_bytes == 0:
            return 1.0
        rank0_bytes = self.per_node_profiles[0].total_bytes
        return rank0_bytes / float(self.unsharded_model_bytes)


class CheckpointMemoryProfiler:
    """
    Computes static and memory-mapped per-node RAM/VRAM footprints before
    distributing weights over Member 1's gRPC network.
    """

    def __init__(self, config: LLMModelConfig) -> None:
        self.config = config

    def profile_cluster(self, world_size: int) -> ClusterMemoryReport:
        """
        Calculate exact byte allocations per worker rank for `world_size` nodes.
        """
        elem_bytes = dtype_byte_size(self.config.torch_dtype)
        expected = self.config.expected_tensor_shapes()
        unsharded_total = self.config.estimate_memory_bytes()

        replicated_bytes = 0
        col_bytes = 0
        row_bytes = 0

        for name, shape in expected.items():
            num_elems = 1
            for d in shape:
                num_elems *= d
            tensor_bytes = num_elems * elem_bytes
            strategy, _ = WeightDistributor.classify_tensor_strategy(name)
            if strategy == PartitionStrategy.REPLICATED or world_size == 1:
                replicated_bytes += tensor_bytes
            elif strategy == PartitionStrategy.COLUMN_PARALLEL:
                col_bytes += tensor_bytes // world_size
            elif strategy == PartitionStrategy.ROW_PARALLEL:
                row_bytes += tensor_bytes // world_size

        total_node_bytes = replicated_bytes + col_bytes + row_bytes
        profiles: Dict[int, NodeMemoryProfile] = {}
        for rank in range(world_size):
            profiles[rank] = NodeMemoryProfile(
                rank=rank,
                world_size=world_size,
                replicated_bytes=replicated_bytes,
                column_parallel_bytes=col_bytes,
                row_parallel_bytes=row_bytes,
                total_bytes=total_node_bytes,
            )

        return ClusterMemoryReport(
            model_name=self.config.model_name_or_path,
            world_size=world_size,
            unsharded_model_bytes=unsharded_total,
            per_node_profiles=profiles,
        )
