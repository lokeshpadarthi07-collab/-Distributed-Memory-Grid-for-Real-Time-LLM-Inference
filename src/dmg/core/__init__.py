"""
Core shared abstractions, interfaces, and exception hierarchy for the
Distributed Memory Grid across all 6 team modules.
"""

from dmg.core.exceptions import (
    ConfigurationError,
    DMGError,
    IntegrityValidationError,
    MemoryMapError,
    SafetensorsFormatError,
    ShardDistributionError,
    ShapeMismatchError,
)
from dmg.core.interfaces import (
    AttentionParallelismProvider,
    FFNParallelismProvider,
    TensorMetadata,
    WeightLoaderProtocol,
    WorkerShardTarget,
)

__all__ = [
    "DMGError",
    "ConfigurationError",
    "SafetensorsFormatError",
    "MemoryMapError",
    "ShapeMismatchError",
    "IntegrityValidationError",
    "ShardDistributionError",
    "TensorMetadata",
    "WorkerShardTarget",
    "WeightLoaderProtocol",
    "AttentionParallelismProvider",
    "FFNParallelismProvider",
]
