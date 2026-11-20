"""
Member 2 – LLM Model Loading & Memory Mapping Module.

Provides configuration validation, Hugging Face Safetensors integration,
zero-copy memory-mapped weight loading, tensor sharding for distributed
nodes, and weight integrity verification.
"""

from dmg.model_loader.page_warmer import (
    CheckpointWarmupSummary,
    MemoryMappedPageWarmer,
    PageWarmupResult,
)
from dmg.model_loader.quantization import (
    QuantizationMode,
    QuantizedTensor,
    WeightQuantizer,
)
from dmg.model_loader.async_streamer import (
    AsyncTensorStreamer,
    StreamedTensorChunk,
)
from dmg.model_loader.config import (
    SUPPORTED_DTYPES,
    ConfigValidator,
    LLMModelConfig,
    dtype_byte_size,
    normalize_dtype,
)
from dmg.model_loader.diagnostics import (
    CheckpointMemoryProfiler,
    ClusterMemoryReport,
    NodeMemoryProfile,
    bf16_words_to_fp32,
    fp32_to_bf16_words,
    to_framework_tensor,
)
from dmg.model_loader.loader import LLMWeightLoader, TransformerLayerWeights
from dmg.model_loader.mmap_loader import MemoryMappedWeightStore, MemoryMapStats
from dmg.model_loader.safetensors_io import (
    SafetensorsHeader,
    SafetensorsIndex,
    parse_safetensors_header,
    save_safetensors_file,
)
from dmg.model_loader.sharding import (
    NodeShardPackage,
    PartitionStrategy,
    TensorShardDescriptor,
    WeightDistributor,
)
from dmg.model_loader.validator import (
    CheckpointValidationReport,
    TensorIntegrityValidator,
    TensorValidationRecord,
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
    "MemoryMappedWeightStore",
    "MemoryMapStats",
    "LLMWeightLoader",
    "TransformerLayerWeights",
    "PartitionStrategy",
    "TensorShardDescriptor",
    "NodeShardPackage",
    "WeightDistributor",
    "TensorValidationRecord",
    "CheckpointValidationReport",
    "TensorIntegrityValidator",
    "CheckpointMemoryProfiler",
    "ClusterMemoryReport",
    "NodeMemoryProfile",
    "bf16_words_to_fp32",
    "fp32_to_bf16_words",
    "to_framework_tensor",
    "AsyncTensorStreamer",
    "StreamedTensorChunk",
    "QuantizationMode",
    "QuantizedTensor",
    "WeightQuantizer",
    "MemoryMappedPageWarmer",
    "PageWarmupResult",
    "CheckpointWarmupSummary",
]
