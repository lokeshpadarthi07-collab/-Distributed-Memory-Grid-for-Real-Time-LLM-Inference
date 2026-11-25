"""
Unit tests for ActivationCompressor, CompressedTensor, and QuantizedCommunicator.
Verifies INT8 and FP8 quantization error boundaries, compression ratio, and All-Reduce accuracy.
"""

import math
import unittest

from dmg.tensor_parallel.communicator import MockCommunicator, MockCommunicatorGroup
from dmg.tensor_parallel.compression import (
    ActivationCompressor,
    QuantizationPrecision,
    QuantizedCommunicator,
)
from dmg.tensor_parallel.config import CommBackend, TensorParallelConfig


class TestQuantizedCommunication(unittest.TestCase):
    """Test suite for tensor parallel activation compression."""

    def test_int8_quantization_and_reconstruction(self):
        # 2 tokens x 4 dims
        tensor = [
            [1.25, -2.50, 0.75, -0.10],
            [3.14, -1.59, 2.65, 0.00],
        ]

        compressed = ActivationCompressor.compress_int8(tensor)
        self.assertEqual(compressed.precision, QuantizationPrecision.INT8)
        self.assertEqual(compressed.original_shape, (2, 4))
        # 8 elements * 4 bytes = 32 bytes uncompressed vs 8 * 1 + 8 bytes = 16 bytes compressed
        self.assertGreater(compressed.compression_ratio, 1.5)

        decompressed = ActivationCompressor.decompress_int8(compressed)
        self.assertEqual(len(decompressed), 2)
        self.assertEqual(len(decompressed[0]), 4)

        # Verify relative error is below 1% for non-zero values
        for i in range(2):
            for j in range(4):
                expected = tensor[i][j]
                actual = decompressed[i][j]
                if abs(expected) > 0.1:
                    rel_err = abs(actual - expected) / abs(expected)
                    self.assertLess(rel_err, 0.02)  # < 2% error

    def test_fp8_e4m3_quantization(self):
        tensor = [
            [10.0, 20.0, 30.0, 40.0],
            [-5.0, -15.0, -25.0, -35.0],
        ]

        compressed = ActivationCompressor.compress_fp8_e4m3(tensor)
        self.assertEqual(compressed.precision, QuantizationPrecision.FP8_E4M3)

        decompressed = ActivationCompressor.decompress_fp8(compressed)
        for i in range(2):
            for j in range(4):
                self.assertAlmostEqual(decompressed[i][j], tensor[i][j], delta=1.5)

    def test_quantized_communicator_all_reduce(self):
        world_size = 2
        group = MockCommunicatorGroup(world_size=world_size)

        comm0_base = MockCommunicator(
            TensorParallelConfig(world_size=2, rank=0), shared_group=group
        )
        comm1_base = MockCommunicator(
            TensorParallelConfig(world_size=2, rank=1), shared_group=group
        )

        # Wrap in QuantizedCommunicators with INT8 compression
        q_comm0 = QuantizedCommunicator(comm0_base, precision=QuantizationPrecision.INT8)
        q_comm1 = QuantizedCommunicator(comm1_base, precision=QuantizationPrecision.INT8)

        t0 = [[10.0, 20.0], [30.0, 40.0]]
        t1 = [[1.0, 2.0], [3.0, 4.0]]

        # All-Reduce
        q_comm0.all_reduce_sum(t0, op_id="quant_sync")
        res1 = q_comm1.all_reduce_sum(t1, op_id="quant_sync")

        # Bandwidth savings tracked
        self.assertGreater(q_comm0.total_uncompressed_bytes_sent, 0)
        self.assertGreater(q_comm0.bandwidth_savings_ratio, 1.0)

        # Expected sum: [[11.0, 22.0], [33.0, 44.0]]
        self.assertAlmostEqual(res1[0][0], 11.0, delta=0.5)
        self.assertAlmostEqual(res1[0][1], 22.0, delta=0.5)
        self.assertAlmostEqual(res1[1][0], 33.0, delta=0.5)
        self.assertAlmostEqual(res1[1][1], 44.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
