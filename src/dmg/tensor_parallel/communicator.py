"""
Distributed communication engine for Tensor Parallelism.
Provides both PyTorch Distributed (`torch.distributed`) collective operations
and deterministic in-memory multi-rank simulation for testing and CPU execution.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dmg.tensor_parallel.config import CommBackend, TensorParallelConfig
from dmg.tensor_parallel.exceptions import CommunicationSyncError
from dmg.tensor_parallel.interfaces import CommunicatorProtocol
from dmg.tensor_parallel.splitting import _concat_dim, _get_shape

logger = logging.getLogger(__name__)


def _add_recursive(a: Any, b: Any) -> Any:
    """Recursively add two nested numeric lists or scalars."""
    if isinstance(a, list) and isinstance(b, list):
        return [_add_recursive(x, y) for x, y in zip(a, b)]
    return a + b


class MockCommunicatorGroup:
    """
    Shared communication bus simulating a multi-node cluster in-memory.
    Ranks exchange tensors via thread-safe/in-memory message registries.
    """

    def __init__(self, world_size: int):
        self.world_size = world_size
        self._registry: Dict[str, List[Any]] = {}
        self.total_bytes_transferred: int = 0
        self.collective_op_count: int = 0

    def register_step(self, op_id: str, rank: int, tensor: Any) -> None:
        """Register a tensor from a specific rank for a collective operation."""
        if op_id not in self._registry:
            self._registry[op_id] = [None] * self.world_size
        self._registry[op_id][rank] = copy.deepcopy(tensor)

    def compute_all_reduce(self, op_id: str) -> Any:
        """Execute in-memory All-Reduce Sum across all registered ranks."""
        if op_id not in self._registry:
            raise CommunicationSyncError(f"Operation {op_id} not found in registry.")

        all_tensors = self._registry[op_id]
        valid_tensors = [t for t in all_tensors if t is not None]
        if not valid_tensors:
            raise CommunicationSyncError(f"No valid tensors registered for op {op_id}")

        self.collective_op_count += 1
        first = valid_tensors[0]

        # PyTorch reduction
        if hasattr(first, "__class__") and "Tensor" in first.__class__.__name__:
            import torch
            reduced = torch.zeros_like(first)
            for t in valid_tensors:
                reduced = reduced + t
            return reduced

        # NumPy reduction
        if hasattr(first, "__class__") and "ndarray" in first.__class__.__name__:
            import numpy as np
            reduced = np.zeros_like(first)
            for t in valid_tensors:
                reduced = reduced + t
            return reduced

        # Python nested lists reduction
        reduced = copy.deepcopy(valid_tensors[0])
        for t in valid_tensors[1:]:
            reduced = _add_recursive(reduced, t)
        return reduced

    def compute_all_gather(self, op_id: str, dim: int = 0) -> Any:
        """Execute in-memory All-Gather concatenating all rank tensors."""
        if op_id not in self._registry:
            raise CommunicationSyncError(f"Operation {op_id} not found in registry.")

        all_tensors = self._registry[op_id]
        valid_tensors = [t for t in all_tensors if t is not None]
        self.collective_op_count += 1
        return _concat_dim(valid_tensors, dim=dim)

    def compute_broadcast(self, op_id: str, src_rank: int = 0) -> Any:
        """Execute in-memory Broadcast from src_rank."""
        if op_id not in self._registry:
            raise CommunicationSyncError(f"Operation {op_id} not found in registry.")

        src_tensor = self._registry[op_id][src_rank]
        if src_tensor is None:
            raise CommunicationSyncError(
                f"Source rank {src_rank} has not registered tensor for {op_id}"
            )
        self.collective_op_count += 1
        return copy.deepcopy(src_tensor)


class MockCommunicator(CommunicatorProtocol):
    """Rank-local communicator communicating via MockCommunicatorGroup."""

    def __init__(
        self,
        config: TensorParallelConfig,
        shared_group: Optional[MockCommunicatorGroup] = None,
    ):
        self._config = config
        self._group = shared_group or MockCommunicatorGroup(config.world_size)
        self._op_counter = 0

    @property
    def rank(self) -> int:
        return self._config.rank

    @property
    def world_size(self) -> int:
        return self._config.world_size

    def _next_op_id(self, op_name: str) -> str:
        self._op_counter += 1
        return f"{op_name}_{self._op_counter}"

    def submit_tensor(self, op_id: str, tensor: Any) -> None:
        """Submit a tensor for a named collective operation."""
        self._group.register_step(op_id, self.rank, tensor)

    def all_reduce_sum(self, tensor: Optional[Any] = None, op_id: Optional[str] = None) -> Any:
        oid = op_id or self._next_op_id("all_reduce")
        if tensor is not None:
            self._group.register_step(oid, self.rank, tensor)
        return self._group.compute_all_reduce(oid)

    def all_gather(
        self, tensor: Optional[Any] = None, dim: int = 0, op_id: Optional[str] = None
    ) -> Any:
        oid = op_id or self._next_op_id("all_gather")
        if tensor is not None:
            self._group.register_step(oid, self.rank, tensor)
        return self._group.compute_all_gather(oid, dim=dim)

    def broadcast(
        self, tensor: Optional[Any] = None, src_rank: int = 0, op_id: Optional[str] = None
    ) -> Any:
        oid = op_id or self._next_op_id("broadcast")
        if tensor is not None:
            self._group.register_step(oid, self.rank, tensor)
        return self._group.compute_broadcast(oid, src_rank=src_rank)


class TorchDistributedCommunicator(CommunicatorProtocol):
    """Production communicator wrapping torch.distributed collectives."""

    def __init__(self, config: TensorParallelConfig):
        self._config = config
        self._initialized = False

    def init_distributed(self) -> None:
        """Initialize PyTorch distributed ProcessGroup if not already initialized."""
        try:
            import torch.distributed as dist

            if not dist.is_initialized():
                backend_str = (
                    "nccl" if self._config.backend == CommBackend.NCCL else "gloo"
                )
                dist.init_process_group(
                    backend=backend_str,
                    init_method=self._config.init_method,
                    world_size=self._config.world_size,
                    rank=self._config.rank,
                )
            self._initialized = True
        except ImportError:
            raise CommunicationSyncError(
                "PyTorch is not installed. Use MockCommunicator for CPU/simulation."
            )
        except Exception as exc:
            raise CommunicationSyncError(f"Failed to initialize torch.distributed: {exc}")

    @property
    def rank(self) -> int:
        return self._config.rank

    @property
    def world_size(self) -> int:
        return self._config.world_size

    def all_reduce_sum(self, tensor: Any) -> Any:
        """Execute torch.distributed.all_reduce with SUM op."""
        try:
            import torch.distributed as dist

            if not dist.is_initialized():
                self.init_distributed()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return tensor
        except Exception as exc:
            raise CommunicationSyncError(f"torch.distributed all_reduce failed: {exc}")

    def all_gather(self, tensor: Any, dim: int = 0) -> Any:
        """Execute torch.distributed.all_gather."""
        try:
            import torch
            import torch.distributed as dist

            if not dist.is_initialized():
                self.init_distributed()
            output_tensors = [torch.empty_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(output_tensors, tensor)
            return torch.cat(output_tensors, dim=dim)
        except Exception as exc:
            raise CommunicationSyncError(f"torch.distributed all_gather failed: {exc}")

    def broadcast(self, tensor: Any, src_rank: int = 0) -> Any:
        """Execute torch.distributed.broadcast."""
        try:
            import torch.distributed as dist

            if not dist.is_initialized():
                self.init_distributed()
            dist.broadcast(tensor, src=src_rank)
            return tensor
        except Exception as exc:
            raise CommunicationSyncError(f"torch.distributed broadcast failed: {exc}")


def create_communicator(
    config: TensorParallelConfig,
    shared_mock_group: Optional[MockCommunicatorGroup] = None,
) -> CommunicatorProtocol:
    """Factory creating appropriate communicator based on configuration backend."""
    if config.backend in (CommBackend.NCCL, CommBackend.GLOO):
        return TorchDistributedCommunicator(config)
    return MockCommunicator(config, shared_group=shared_mock_group)
