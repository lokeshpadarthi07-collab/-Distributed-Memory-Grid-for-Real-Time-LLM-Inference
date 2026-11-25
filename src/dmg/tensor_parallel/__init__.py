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
from dmg.tensor_parallel.compression import (
    ActivationCompressor,
    CompressedTensor,
    QuantizationPrecision,
    QuantizedCommunicator,
)
from dmg.tensor_parallel.config import (
    AttentionType,
    CommBackend,
    ParallelAttentionConfig,
    TensorParallelConfig,
)
from dmg.tensor_parallel.engine import DistributedInferenceEngine
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
from dmg.tensor_parallel.kv_cache import (
    DistributedKVCacheManager,
    RankKVCacheState,
)
from dmg.tensor_parallel.memory_efficient_attention import (
    ChunkedAttentionKernel,
    OnlineSoftmaxAccumulator,
)
from dmg.tensor_parallel.paged_cache import (
    BlockTable,
    PagedKVCacheManager,
    PhysicalBlock,
)
from dmg.tensor_parallel.profiler import (
    ProfilingRecord,
    TensorParallelProfiler,
)
from dmg.tensor_parallel.rotary import (
    GQAHeadExpander,
    RotaryEmbedding,
)
from dmg.tensor_parallel.sharding import (
    AttentionHeadSharder,
    AttentionShardManifest,
    HeadPartition,
)
from dmg.tensor_parallel.splitting import TensorMerger, TensorSplitter
from dmg.tensor_parallel.synchronizer import (
    AttentionOutputSynchronizer,
    TensorParallelAttentionBlock,
)

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
    "AttentionOutputSynchronizer",
    "TensorParallelAttentionBlock",
    "DistributedKVCacheManager",
    "RankKVCacheState",
    "TensorParallelProfiler",
    "ProfilingRecord",
    "DistributedInferenceEngine",
    "RotaryEmbedding",
    "GQAHeadExpander",
    "ChunkedAttentionKernel",
    "OnlineSoftmaxAccumulator",
    "PhysicalBlock",
    "BlockTable",
    "PagedKVCacheManager",
    "QuantizationPrecision",
    "CompressedTensor",
    "ActivationCompressor",
    "QuantizedCommunicator",
]
