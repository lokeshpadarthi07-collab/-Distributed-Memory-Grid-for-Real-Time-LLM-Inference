"""
Member 2 – LLM Model Loading & Memory Mapping Module.

Provides configuration validation, Hugging Face Safetensors integration,
zero-copy memory-mapped weight loading, tensor sharding for distributed
nodes, and weight integrity verification.
"""

from dmg.model_loader.config import (
    SUPPORTED_DTYPES,
    ConfigValidator,
    LLMModelConfig,
    dtype_byte_size,
    normalize_dtype,
)
from dmg.model_loader.safetensors_io import (
    SafetensorsHeader,
    SafetensorsIndex,
    parse_safetensors_header,
    save_safetensors_file,
)

__all__ = [
    "LLMModelConfig",
    "ConfigValidator",
    "SUPPORTED_DTYPES",
    "normalize_dtype",
    "dtype_byte_size",
    "SafetensorsHeader",
    "SafetensorsIndex",
    "parse_safetensors_header",
    "save_safetensors_file",
]
