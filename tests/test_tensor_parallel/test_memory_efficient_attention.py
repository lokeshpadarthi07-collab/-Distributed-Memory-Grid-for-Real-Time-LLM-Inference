"""
Unit and parity tests for OnlineSoftmaxAccumulator and ChunkedAttentionKernel.
Verifies numerical equivalence between chunked tiled attention and monolithic attention.
"""

import math
import unittest

from dmg.tensor_parallel.attention import (
    ColumnParallelLinear,
    ParallelSelfAttention,
    RowParallelLinear,
)
from dmg.tensor_parallel.config import ParallelAttentionConfig, TensorParallelConfig
from dmg.tensor_parallel.memory_efficient_attention import (
    ChunkedAttentionKernel,
    OnlineSoftmaxAccumulator,
)


class TestMemoryEfficientAttention(unittest.TestCase):
    """Test suite for online softmax and chunked attention."""

    def test_online_softmax_single_vector(self):
        head_dim = 2
        acc = OnlineSoftmaxAccumulator(head_dim=head_dim)

        # Tile 1: 2 keys with scores [2.0, 1.0]
        v_tile_1 = [[1.0, 0.0], [0.0, 1.0]]
        acc.update([2.0, 1.0], v_tile_1)

        # Tile 2: 2 keys with scores [3.0, 0.0]
        v_tile_2 = [[2.0, 2.0], [0.5, 0.5]]
        acc.update([3.0, 0.0], v_tile_2)

        result = acc.finalize()

        # Compare with standard softmax over [2.0, 1.0, 3.0, 0.0]
        scores = [2.0, 1.0, 3.0, 0.0]
        max_s = max(scores)
        exps = [math.exp(s - max_s) for s in scores]
        sum_e = sum(exps)
        weights = [e / sum_e for e in exps]
        all_v = v_tile_1 + v_tile_2

        expected = [
            sum(weights[i] * all_v[i][d] for i in range(4)) for d in range(head_dim)
        ]

        self.assertAlmostEqual(result[0], expected[0], places=5)
        self.assertAlmostEqual(result[1], expected[1], places=5)

    def test_chunked_attention_parity(self):
        # Q: 4 tokens x 2 dims
        # K, V: 4 tokens x 2 dims
        q = [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.2, 0.8],
            [0.0, 1.0],
        ]
        k = [
            [1.0, 0.1],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.1, 0.9],
        ]
        v = [
            [2.0, 1.0],
            [1.5, 0.5],
            [0.5, 2.0],
            [0.0, 3.0],
        ]

        # Block size = 2 (triggers multi-chunk execution)
        out_chunked = ChunkedAttentionKernel.chunked_scaled_dot_product(
            q, k, v, block_size=2, causal_mask=True
        )

        # Single block (all in one tile)
        out_single_block = ChunkedAttentionKernel.chunked_scaled_dot_product(
            q, k, v, block_size=16, causal_mask=True
        )

        self.assertEqual(len(out_chunked), 4)
        for t in range(4):
            for d in range(2):
                self.assertAlmostEqual(
                    out_chunked[t][d], out_single_block[t][d], places=5
                )

    def test_sliding_window_masking(self):
        q = [[1.0, 1.0]] * 6
        k = [[1.0, 1.0]] * 6
        v = [[float(i), 0.0] for i in range(6)]

        # Window = 2: token i can only attend to [i-2, i]
        out_windowed = ChunkedAttentionKernel.chunked_scaled_dot_product(
            q, k, v, block_size=2, causal_mask=True, sliding_window=2
        )
        self.assertEqual(len(out_windowed), 6)

    def test_parallel_self_attention_with_chunked_kernel(self):
        tp_cfg = TensorParallelConfig(world_size=2, rank=0)
        att_cfg = ParallelAttentionConfig(
            hidden_size=4,
            num_attention_heads=2,
            head_dim=2,
            use_chunked_attention=True,
            chunk_size=2,
            tp_config=tp_cfg,
        )

        w_q = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        w_k = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        w_v = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        w_o = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]

        q_proj = ColumnParallelLinear(w_q)
        k_proj = ColumnParallelLinear(w_k)
        v_proj = ColumnParallelLinear(w_v)
        o_proj = RowParallelLinear(w_o)

        attention = ParallelSelfAttention(att_cfg, q_proj, k_proj, v_proj, o_proj)

        x = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
        ]
        out_local = attention.forward_local(x, causal_mask=True)
        self.assertEqual(len(out_local), 4)
        self.assertEqual(len(out_local[0]), 4)


if __name__ == "__main__":
    unittest.main()
