"""
Unit tests for TensorSplitter and TensorMerger.
Tests column, row, and attention-head splitting and round-trip reconstruction.
"""

import unittest

from dmg.tensor_parallel.exceptions import (
    AttentionHeadSplitError,
    ParallelShapeMismatchError,
)
from dmg.tensor_parallel.splitting import TensorMerger, TensorSplitter


class Dummy2DTensor:
    """Mock 2D tensor supporting shape and slicing for testing without PyTorch."""

    def __init__(self, data):
        self.data = data
        self.shape = (len(data), len(data[0]) if data else 0)

    def __getitem__(self, idx):
        row_slice, col_slice = idx
        sliced_rows = self.data[row_slice]
        return Dummy2DTensor([row[col_slice] for row in sliced_rows])

    def to_list(self):
        return self.data


class TestTensorSplittingAndMerging(unittest.TestCase):
    """Test suite for tensor splitting and merging operators."""

    def setUp(self):
        # 4x6 matrix: 4 rows (e.g. out_features), 6 cols (e.g. in_features)
        self.matrix_data = [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
        ]
        self.matrix = Dummy2DTensor(self.matrix_data)

    def test_split_column(self):
        # World size = 2, splitting along dim 0 (rows/out_features)
        s0 = TensorSplitter.split_column(self.matrix, world_size=2, rank=0, dim=0)
        s1 = TensorSplitter.split_column(self.matrix, world_size=2, rank=1, dim=0)

        self.assertEqual(s0.shape, (2, 6))
        self.assertEqual(s1.shape, (2, 6))
        self.assertEqual(s0.to_list(), self.matrix_data[0:2])
        self.assertEqual(s1.to_list(), self.matrix_data[2:4])

    def test_split_row(self):
        # World size = 3, splitting along dim 1 (cols/in_features)
        s0 = TensorSplitter.split_row(self.matrix, world_size=3, rank=0, dim=1)
        s1 = TensorSplitter.split_row(self.matrix, world_size=3, rank=1, dim=1)
        s2 = TensorSplitter.split_row(self.matrix, world_size=3, rank=2, dim=1)

        self.assertEqual(s0.shape, (4, 2))
        self.assertEqual(s1.shape, (4, 2))
        self.assertEqual(s2.shape, (4, 2))

    def test_split_attention_heads(self):
        # 8 heads, head_dim = 4 -> total dim 32
        # Mock weight matrix: 32 x 16
        head_data = [[i * 16 + j for j in range(16)] for i in range(32)]
        weight = Dummy2DTensor(head_data)

        # Split across world_size = 4 (2 heads per rank, 2*4 = 8 rows per rank)
        slices = [
            TensorSplitter.split_attention_heads(
                weight, num_heads=8, head_dim=4, world_size=4, rank=r, dim=0
            )
            for r in range(4)
        ]

        for r, s in enumerate(slices):
            self.assertEqual(s.shape, (8, 16))
            self.assertEqual(s.to_list(), head_data[r * 8 : (r + 1) * 8])

    def test_split_indivisible_dim(self):
        with self.assertRaises(ParallelShapeMismatchError):
            # 4 rows not divisible by 3 ranks
            TensorSplitter.split_along_dim(self.matrix, dim=0, world_size=3, rank=0)

    def test_split_heads_indivisible(self):
        weight = Dummy2DTensor([[1] * 16] * 30)
        with self.assertRaises(AttentionHeadSplitError):
            # 7 heads not divisible by 4 ranks
            TensorSplitter.split_attention_heads(
                weight, num_heads=7, head_dim=4, world_size=4, rank=0
            )

    def test_merge_native_lists(self):
        chunk0 = [[1, 2], [3, 4]]
        chunk1 = [[5, 6], [7, 8]]
        merged_row = TensorMerger.merge_along_dim([chunk0, chunk1], dim=0)
        self.assertEqual(merged_row, [[1, 2], [3, 4], [5, 6], [7, 8]])

        merged_col = TensorMerger.merge_along_dim([chunk0, chunk1], dim=1)
        self.assertEqual(merged_col, [[1, 2, 5, 6], [3, 4, 7, 8]])


if __name__ == "__main__":
    unittest.main()
