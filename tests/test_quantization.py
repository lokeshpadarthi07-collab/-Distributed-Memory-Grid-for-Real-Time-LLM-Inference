"""
Unit tests for INT8 and INT4 weight quantization and dequantization (`quantization.py`).
"""

from __future__ import annotations

import numpy as np

from dmg.model_loader.quantization import QuantizationMode, WeightQuantizer


class TestWeightQuantization:
    """Tests for symmetric per-tensor, per-channel INT8, and blockwise INT4 quantization."""

    def test_int8_per_tensor_quantization(self) -> None:
        rng = np.random.default_rng(42)
        weights = rng.standard_normal((32, 64)).astype(np.float32)

        qtensor = WeightQuantizer.quantize_int8_per_tensor("test_proj", weights)
        assert qtensor.quant_dtype == "int8"
        assert qtensor.mode == QuantizationMode.INT8_PER_TENSOR
        assert qtensor.qdata.shape == weights.shape

        reconstructed = WeightQuantizer.dequantize_int8_per_tensor(qtensor)
        assert reconstructed.shape == weights.shape

        # Quantization SNR check
        cos_sim = np.dot(weights.flat, reconstructed.flat) / (
            np.linalg.norm(weights) * np.linalg.norm(reconstructed)
        )
        assert cos_sim > 0.99

    def test_int8_per_channel_quantization(self) -> None:
        rng = np.random.default_rng(123)
        weights = rng.standard_normal((16, 32)).astype(np.float32)

        qtensor = WeightQuantizer.quantize_int8_per_channel("gate_proj", weights)
        assert qtensor.mode == QuantizationMode.INT8_PER_CHANNEL
        assert qtensor.scales.shape == (16, 1)

        reconstructed = WeightQuantizer.dequantize_int8_per_channel(qtensor)
        assert reconstructed.shape == weights.shape
        cos_sim = np.dot(weights.flat, reconstructed.flat) / (
            np.linalg.norm(weights) * np.linalg.norm(reconstructed)
        )
        assert cos_sim > 0.995

    def test_int4_blockwise_quantization(self) -> None:
        rng = np.random.default_rng(999)
        weights = rng.standard_normal((32, 32)).astype(np.float32)

        qtensor = WeightQuantizer.quantize_int4_blockwise("q_proj", weights, block_size=32)
        assert qtensor.mode == QuantizationMode.INT4_BLOCKWISE
        # Packed 2 nibbles per uint8 byte -> size is halved
        assert qtensor.qdata.nbytes == (weights.size // 2)

        reconstructed = WeightQuantizer.dequantize_int4_blockwise(qtensor)
        assert reconstructed.shape == weights.shape
        cos_sim = np.dot(weights.flat, reconstructed.flat) / (
            np.linalg.norm(weights) * np.linalg.norm(reconstructed)
        )
        assert cos_sim > 0.96
