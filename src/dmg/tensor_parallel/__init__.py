"""
Distributed Memory Grid - Member 3: Tensor Parallelism (Attention Heads).
"""

from __future__ import annotations

from dmg.tensor_parallel.attention import (
    ColumnParallelLinear,
    ParallelSelfAttention,
    RowParallelLinear,
)
from dmg.tensor_parallel.communicator import (
    MockCommunicator,
    MockCommunicatorGroup,
    TorchDistributedCommunicator,
    create_communicator,
)
from dmg.tensor_parallel.config import (
    AttentionType,
    CommBackend,
    ParallelAttentionConfig,
    TensorParallelConfig,
)
from dmg.tensor_parallel.exceptions import (
    AttentionHeadSplitError,
    CommunicationSyncError,
    HeadReconstructionError,
    InvalidParallelConfigError,
    ParallelShapeMismatchError,
    TensorParallelError,
)
from dmg.tensor_parallel.interfaces import (
    CommunicatorProtocol,
    ParallelAttentionProtocol,
    TensorSplitterProtocol,
)
from dmg.tensor_parallel.sharding import (
    AttentionHeadSharder,
    AttentionShardManifest,
    HeadPartition,
)
from dmg.tensor_parallel.splitting import TensorMerger, TensorSplitter

__all__ = [
    "AttentionType",
    "CommBackend",
    "TensorParallelConfig",
    "ParallelAttentionConfig",
    "TensorParallelError",
    "InvalidParallelConfigError",
    "AttentionHeadSplitError",
    "ParallelShapeMismatchError",
    "CommunicationSyncError",
    "HeadReconstructionError",
    "CommunicatorProtocol",
    "TensorSplitterProtocol",
    "ParallelAttentionProtocol",
    "TensorSplitter",
    "TensorMerger",
    "MockCommunicator",
    "MockCommunicatorGroup",
    "TorchDistributedCommunicator",
    "create_communicator",
    "AttentionHeadSharder",
    "HeadPartition",
    "AttentionShardManifest",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "ParallelSelfAttention",
]
