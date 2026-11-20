"""
Low-bit INT8 / INT4 weight quantization, scale calibration, and dequantization
engine for memory-efficient LLM inference in the Distributed Memory Grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np

from dmg.core.exceptions import IntegrityValidationError


class QuantizationMode(str, Enum):
    """Supported quantization precision modes."""

    INT8_PER_TENSOR = "int8_per_tensor"
    INT8_PER_CHANNEL = "int8_per_channel"
    INT4_BLOCKWISE = "int4_blockwise"


@dataclass(frozen=True)
class QuantizedTensor:
    """Represents a quantized parameter tensor with associated scaling factors and zero points."""

    tensor_name: str
    original_shape: Tuple[int, ...]
    quant_dtype: str
    qdata: np.ndarray[Any, Any]
    scales: np.ndarray[Any, Any]
    zero_points: Optional[np.ndarray[Any, Any]]
    mode: QuantizationMode
    block_size: Optional[int] = None

    @property
    def compressed_bytes(self) -> int:
        """Memory footprint of quantized data plus metadata scales."""
        scales_bytes = self.scales.nbytes
        zp_bytes = self.zero_points.nbytes if self.zero_points is not None else 0
        return self.qdata.nbytes + scales_bytes + zp_bytes


class WeightQuantizer:
    """
    Symmetric and asymmetric weight quantizer for LLM weights.
    Reduces memory footprint by 2x (INT8) or 4x (INT4) while maintaining
    high numerical fidelity for distributed worker nodes.
    """

    @staticmethod
    def quantize_int8_per_tensor(
        tensor_name: str,
        arr: np.ndarray[Any, Any],
    ) -> QuantizedTensor:
        """Symmetric per-tensor INT8 quantization."""
        if not np.issubdtype(arr.dtype, np.floating):
            raise IntegrityValidationError(
                f"Cannot quantize non-floating point tensor '{tensor_name}' of dtype {arr.dtype}"
            )

        max_val = float(np.max(np.abs(arr)))
        scale = max_val / 127.0 if max_val > 0.0 else 1.0

        qdata = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)

        return QuantizedTensor(
            tensor_name=tensor_name,
            original_shape=tuple(arr.shape),
            quant_dtype="int8",
            qdata=qdata,
            scales=np.array([scale], dtype=np.float32),
            zero_points=None,
            mode=QuantizationMode.INT8_PER_TENSOR,
        )

    @staticmethod
    def dequantize_int8_per_tensor(qtensor: QuantizedTensor) -> np.ndarray[Any, Any]:
        """Dequantize an INT8 per-tensor quantized tensor back to float32."""
        scale = float(qtensor.scales[0])
        return (qtensor.qdata.astype(np.float32) * scale).reshape(qtensor.original_shape)

    @staticmethod
    def quantize_int8_per_channel(
        tensor_name: str,
        arr: np.ndarray[Any, Any],
        axis: int = 0,
    ) -> QuantizedTensor:
        """
        Symmetric per-channel INT8 quantization along the output feature axis.
        Provides significantly higher accuracy than per-tensor quantization for linear projections.
        """
        if arr.ndim != 2:
            return WeightQuantizer.quantize_int8_per_tensor(tensor_name, arr)

        # Compute max along row/column axis
        abs_max = np.max(np.abs(arr), axis=1 if axis == 0 else 0, keepdims=True)
        scales = np.where(abs_max > 0.0, abs_max / 127.0, 1.0).astype(np.float32)

        qdata = np.clip(np.round(arr / scales), -128, 127).astype(np.int8)

        return QuantizedTensor(
            tensor_name=tensor_name,
            original_shape=tuple(arr.shape),
            quant_dtype="int8",
            qdata=qdata,
            scales=scales,
            zero_points=None,
            mode=QuantizationMode.INT8_PER_CHANNEL,
        )

    @staticmethod
    def dequantize_int8_per_channel(qtensor: QuantizedTensor) -> np.ndarray[Any, Any]:
        """Dequantize a per-channel INT8 quantized tensor back to float32."""
        return (qtensor.qdata.astype(np.float32) * qtensor.scales).reshape(qtensor.original_shape)

    @staticmethod
    def quantize_int4_blockwise(
        tensor_name: str,
        arr: np.ndarray[Any, Any],
        block_size: int = 32,
    ) -> QuantizedTensor:
        """
        Blockwise INT4 quantization with packed uint8 storage (2 nibbles per byte).
        Delivers 4x memory compression for large Feed-Forward and Attention weight matrices.
        """
        flat = arr.astype(np.float32).reshape(-1)
        num_elements = flat.shape[0]

        # Pad to multiple of block_size if necessary
        pad_len = (block_size - (num_elements % block_size)) % block_size
        if pad_len > 0:
            flat = np.pad(flat, (0, pad_len), mode="constant")

        blocks = flat.reshape(-1, block_size)
        abs_max = np.max(np.abs(blocks), axis=1, keepdims=True)
        scales = np.where(abs_max > 0.0, abs_max / 7.0, 1.0).astype(np.float32)

        # Symmetric [-7, 7] signed nibbles
        q_nibbles = np.clip(np.round(blocks / scales), -7, 7).astype(np.int8)

        # Shift to unsigned [0..14] for clean 4-bit packing
        u4 = (q_nibbles + 8).astype(np.uint8)
        u4_flat = u4.reshape(-1)

        # Pack 2 nibbles per uint8 byte
        even = u4_flat[0::2]
        odd = u4_flat[1::2]
        packed = (even | (odd << 4)).astype(np.uint8)

        return QuantizedTensor(
            tensor_name=tensor_name,
            original_shape=tuple(arr.shape),
            quant_dtype="int4_packed",
            qdata=packed,
            scales=scales,
            zero_points=None,
            mode=QuantizationMode.INT4_BLOCKWISE,
            block_size=block_size,
        )

    @staticmethod
    def dequantize_int4_blockwise(qtensor: QuantizedTensor) -> np.ndarray[Any, Any]:
        """Unpack and dequantize blockwise INT4 back to original float32 shape."""
        packed = qtensor.qdata
        low_nibbles = (packed & 0x0F).astype(np.int8) - 8
        high_nibbles = ((packed >> 4) & 0x0F).astype(np.int8) - 8

        unpacked = np.empty(packed.size * 2, dtype=np.int8)
        unpacked[0::2] = low_nibbles
        unpacked[1::2] = high_nibbles

        assert qtensor.block_size is not None
        blocks = unpacked.reshape(-1, qtensor.block_size)
        dequant_blocks = blocks.astype(np.float32) * qtensor.scales

        orig_num_elements = 1
        for d in qtensor.original_shape:
            orig_num_elements *= d

        flat_trimmed = dequant_blocks.reshape(-1)[:orig_num_elements]
        return flat_trimmed.reshape(qtensor.original_shape)
