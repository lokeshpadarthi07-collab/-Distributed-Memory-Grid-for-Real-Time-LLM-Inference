"""
Unit tests for BF16/FP32 bit conversions, framework tensor bridging,
and cluster memory profiling (`diagnostics.py`).
"""

from __future__ import annotations

import numpy as np

from dmg.model_loader.config import LLMModelConfig
from dmg.model_loader.diagnostics import (
    CheckpointMemoryProfiler,
    bf16_words_to_fp32,
    fp32_to_bf16_words,
    to_framework_tensor,
)


class TestDiagnosticsAndBridge:
    """Tests for `CheckpointMemoryProfiler` and BF16 conversion utilities."""

    def test_bf16_fp32_roundtrip_conversion(self) -> None:
        original = np.array([0.0, 1.0, -2.5, 3.140625, 128.0], dtype=np.float32)
        bf16_words = fp32_to_bf16_words(original)
        assert bf16_words.dtype == np.uint16
        reconstructed = bf16_words_to_fp32(bf16_words)
        assert reconstructed.dtype == np.float32
        assert np.allclose(original, reconstructed, rtol=1e-2)

    def test_cluster_memory_profiler_savings(
        self, tiny_model_config: LLMModelConfig
    ) -> None:
        profiler = CheckpointMemoryProfiler(tiny_model_config)
        report_1 = profiler.profile_cluster(world_size=1)
        report_4 = profiler.profile_cluster(world_size=4)

        assert report_1.per_node_profiles[0].total_bytes == report_1.unsharded_model_bytes
        assert report_4.per_node_profiles[0].total_bytes < report_1.unsharded_model_bytes
        assert 0.0 < report_4.memory_savings_ratio < 1.0

    def test_framework_tensor_bridge(self) -> None:
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        bridged = to_framework_tensor(arr, copy=True)
        assert tuple(bridged.shape) == (3, 4)
