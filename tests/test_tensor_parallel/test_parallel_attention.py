"""
Unit tests for ColumnParallelLinear, RowParallelLinear, and ParallelSelfAttention.
"""

import unittest

from dmg.tensor_parallel.attention import (
    ColumnParallelLinear,
    ParallelSelfAttention,
    RowParallelLinear,
)
from dmg.tensor_parallel.config import ParallelAttentionConfig, TensorParallelConfig


class TestParallelAttention(unittest.TestCase):
    """Test suite for distributed attention computation."""

    def test_column_parallel_linear(self):
        # in_features = 4, out_features_shard = 2
        # W_shard: 2 x 4
        w_shard = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
        b_shard = [0.5, -0.5]
        col_linear = ColumnParallelLinear(w_shard, b_shard)

        # Input: 2 tokens x 4 dims
        x = [
            [2.0, 3.0, 0.0, 0.0],
            [1.0, 4.0, 0.0, 0.0],
        ]
        out = col_linear.forward(x)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0]), 2)
        # Token 0: [2.0*1 + 0.5, 3.0*1 - 0.5] = [2.5, 2.5]
        self.assertAlmostEqual(out[0][0], 2.5)
        self.assertAlmostEqual(out[0][1], 2.5)

    def test_row_parallel_linear(self):
        # in_features_shard = 2, out_features = 3
        # W_shard: 3 x 2
        w_shard = [
            [1.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
        row_linear = RowParallelLinear(w_shard)
        # Input: 1 token x 2 dims
        x_local = [[3.0, 4.0]]
        out = row_linear.forward_local(x_local)
        # [3*1 + 4*1, 3*2 + 4*0, 3*0 + 4*2] = [7.0, 6.0, 8.0]
        self.assertEqual(out, [[7.0, 6.0, 8.0]])

    def test_parallel_self_attention_local_forward(self):
        # 2 heads total, world_size = 2 -> 1 local head per rank. head_dim = 2 -> hidden_size = 4
        tp_cfg = TensorParallelConfig(world_size=2, rank=0)
        att_cfg = ParallelAttentionConfig(
            hidden_size=4,
            num_attention_heads=2,
            head_dim=2,
            tp_config=tp_cfg,
        )

        # Identity projections for local head (2 x 4)
        w_q = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        w_k = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        w_v = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        # Row-parallel output projection (4 x 2)
        w_o = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]

        q_proj = ColumnParallelLinear(w_q)
        k_proj = ColumnParallelLinear(w_k)
        v_proj = ColumnParallelLinear(w_v)
        o_proj = RowParallelLinear(w_o)

        attention = ParallelSelfAttention(att_cfg, q_proj, k_proj, v_proj, o_proj)

        # Input: 2 tokens x 4 dims
        x = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
        out_local = attention.forward_local(x, causal_mask=True)
        # Output should be [2 tokens, 4 dims]
        self.assertEqual(len(out_local), 2)
        self.assertEqual(len(out_local[0]), 4)


if __name__ == "__main__":
    unittest.main()
