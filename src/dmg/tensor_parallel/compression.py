"""
Activation quantization and communication compression engine for Tensor Parallelism.
Compresses cross-rank All-Reduce payloads to INT8/FP8 to alleviate interconnect bandwidth
bottlenecks during distributed attention output synchronization.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from dmg.tensor_parallel.exceptions import TensorParallelError
from dmg.tensor_parallel.interfaces import CommunicatorProtocol
from dmg.tensor_parallel.splitting import _get_shape


class QuantizationPrecision(str, Enum):
    """Supported communication compression precisions."""

    FP32 = "fp32"
    INT8 = "int8"
    FP8_E4M3 = "fp8_e4m3"


@dataclass(frozen=True)
class CompressedTensor:
    """Compressed tensor payload with scaling factor for cross-rank transmission."""

    quantized_data: Any
    scale: float
    precision: QuantizationPrecision
    original_shape: Tuple[int, ...]
    uncompressed_bytes: int
    compressed_bytes: int

    @property
    def compression_ratio(self) -> float:
        """Ratio of uncompressed size to compressed wire size."""
        if self.compressed_bytes == 0:
            return 1.0
        return self.uncompressed_bytes / self.compressed_bytes


class ActivationCompressor:
    """Performs dynamic activation quantization and lossless-scale reconstruction."""

    @staticmethod
    def compress_int8(tensor: List[List[float]]) -> CompressedTensor:
        """
        Symmetric per-tensor INT8 quantization.
        Scales values to [-127, 127] signed 8-bit integers.
        """
        shape = _get_shape(tensor)
        flat = []
        for row in tensor:
            flat.extend(row)

        if not flat:
            return CompressedTensor([], 1.0, QuantizationPrecision.INT8, shape, 0, 0)

        max_val = max(abs(x) for x in flat)
        scale = max_val / 127.0 if max_val > 1e-8 else 1.0

        quantized_rows = []
        for row in tensor:
            q_row = [
                int(max(-127, min(127, round(x / scale)))) for x in row
            ]
            quantized_rows.append(q_row)

        num_elements = len(flat)
        uncompressed_bytes = num_elements * 4  # float32 = 4 bytes
        compressed_bytes = num_elements * 1 + 8  # int8 = 1 byte + 8 bytes float64 scale

        return CompressedTensor(
            quantized_data=quantized_rows,
            scale=scale,
            precision=QuantizationPrecision.INT8,
            original_shape=shape,
            uncompressed_bytes=uncompressed_bytes,
            compressed_bytes=compressed_bytes,
        )

    @staticmethod
    def decompress_int8(compressed: CompressedTensor) -> List[List[float]]:
        """Dequantize INT8 integers back to floating-point approximations."""
        scale = compressed.scale
        out = []
        for row in compressed.quantized_data:
            out.append([float(q) * scale for q in row])
        return out

    @staticmethod
    def compress_fp8_e4m3(tensor: List[List[float]]) -> CompressedTensor:
        """
        Simulated FP8 E4M3 quantization (max finite = 448.0, 3 bits of mantissa).
        Offers ~4x bandwidth savings with higher dynamic range than INT8.
        """
        shape = _get_shape(tensor)
        flat = [x for row in tensor for x in row]
        if not flat:
            return CompressedTensor([], 1.0, QuantizationPrecision.FP8_E4M3, shape, 0, 0)

        max_val = max(abs(x) for x in flat)
        max_fp8 = 448.0
        scale = max_val / max_fp8 if max_val > 1e-8 else 1.0

        quantized_rows = []
        step = max_fp8 / 128.0  # Simulated quant step
        for row in tensor:
            q_row = [
                round((x / scale) / step) * step for x in row
            ]
            quantized_rows.append(q_row)

        num_elements = len(flat)
        uncompressed_bytes = num_elements * 4
        compressed_bytes = num_elements * 1 + 8

        return CompressedTensor(
            quantized_data=quantized_rows,
            scale=scale,
            precision=QuantizationPrecision.FP8_E4M3,
            original_shape=shape,
            uncompressed_bytes=uncompressed_bytes,
            compressed_bytes=compressed_bytes,
        )

    @staticmethod
    def decompress_fp8(compressed: CompressedTensor) -> List[List[float]]:
        """Dequantize FP8 values back to standard floats."""
        scale = compressed.scale
        out = []
        for row in compressed.quantized_data:
            out.append([x * scale for x in row])
        return out


class QuantizedCommunicator(CommunicatorProtocol):
    """
    Decorator around CommunicatorProtocol that compresses activations
    before transmission to reduce All-Reduce communication latency.
    """

    def __init__(
        self,
        base_communicator: CommunicatorProtocol,
        precision: QuantizationPrecision = QuantizationPrecision.INT8,
    ):
        self.base_communicator = base_communicator
        self.precision = precision
        self.total_uncompressed_bytes_sent = 0
        self.total_compressed_bytes_sent = 0

    @property
    def rank(self) -> int:
        return self.base_communicator.rank

    @property
    def world_size(self) -> int:
        return self.base_communicator.world_size

    @property
    def bandwidth_savings_ratio(self) -> float:
        if self.total_compressed_bytes_sent == 0:
            return 1.0
        return self.total_uncompressed_bytes_sent / self.total_compressed_bytes_sent

    def all_reduce_sum(self, tensor: Any, op_id: Optional[str] = None) -> Any:
        """Compress payload, execute collective, and dequantize result."""
        if not isinstance(tensor, list):
            # Fallback for unhandled tensor types
            return self.base_communicator.all_reduce_sum(tensor)

        # 1. Compress local payload
        if self.precision == QuantizationPrecision.INT8:
            compressed = ActivationCompressor.compress_int8(tensor)
        elif self.precision == QuantizationPrecision.FP8_E4M3:
            compressed = ActivationCompressor.compress_fp8_e4m3(tensor)
        else:
            return self.base_communicator.all_reduce_sum(tensor)

        self.total_uncompressed_bytes_sent += compressed.uncompressed_bytes
        self.total_compressed_bytes_sent += compressed.compressed_bytes

        # 2. Decompress and perform All-Reduce (or submit to mock group)
        decompressed_local = (
            ActivationCompressor.decompress_int8(compressed)
            if self.precision == QuantizationPrecision.INT8
            else ActivationCompressor.decompress_fp8(compressed)
        )

        if hasattr(self.base_communicator, "submit_tensor") and op_id:
            self.base_communicator.submit_tensor(op_id, decompressed_local)
            return self.base_communicator.all_reduce_sum(op_id=op_id)

        return self.base_communicator.all_reduce_sum(decompressed_local)

    def all_gather(self, tensor: Any, dim: int = 0, op_id: Optional[str] = None) -> Any:
        return self.base_communicator.all_gather(tensor, dim=dim)

    def broadcast(self, tensor: Any, src_rank: int = 0, op_id: Optional[str] = None) -> Any:
        return self.base_communicator.broadcast(tensor, src_rank=src_rank)
