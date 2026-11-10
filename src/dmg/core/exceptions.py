"""
Enterprise exception hierarchy for the Distributed Memory Grid.
"""

from __future__ import annotations


class DMGError(Exception):
    """Base exception for all Distributed Memory Grid errors."""


class ConfigurationError(DMGError):
    """Raised when model or cluster configuration is invalid or inconsistent."""


class SafetensorsFormatError(DMGError):
    """Raised when a Safetensors file or index is malformed or unreadable."""


class MemoryMapError(DMGError):
    """Raised when OS memory mapping or zero-copy slicing fails."""


class ShapeMismatchError(DMGError):
    """Raised when a loaded tensor's shape or dtype does not match the model specification."""


class IntegrityValidationError(DMGError):
    """Raised when tensor weights fail numerical (NaN/Inf) or checksum integrity validation."""


class ShardDistributionError(DMGError):
    """Raised when preparing or partitioning model weights across worker nodes fails."""
