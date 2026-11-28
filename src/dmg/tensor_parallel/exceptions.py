"""
Custom exception hierarchy for Tensor Parallelism and Distributed Attention.
"""

from __future__ import annotations

from dmg.core.exceptions import DMGError


class TensorParallelError(DMGError):
    """Base exception for all Tensor Parallelism errors."""


class InvalidParallelConfigError(TensorParallelError):
    """Raised when tensor parallelism or attention configuration parameters are invalid."""


class AttentionHeadSplitError(TensorParallelError):
    """Raised when attention heads cannot be evenly or validly split across ranks."""


class ParallelShapeMismatchError(TensorParallelError):
    """Raised when tensor shapes fail to match expectations for parallel projection."""


class CommunicationSyncError(TensorParallelError):
    """Raised when collective synchronization (All-Reduce, All-Gather) fails across ranks."""


class HeadReconstructionError(TensorParallelError):
    """Raised when merging or un-sharding attention heads fails to reproduce original tensor."""
