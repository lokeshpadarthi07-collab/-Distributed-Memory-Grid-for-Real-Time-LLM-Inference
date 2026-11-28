"""
Unit tests for parallel Rotary Position Embeddings (RoPE) and GQA head expansion.
"""

import math
import unittest

from dmg.tensor_parallel.attention import (
    ColumnParallelLinear,
    ParallelSelfAttention,
    RowParallelLinear,
)
from dmg.tensor_parallel.config import (
    AttentionType,
    ParallelAttentionConfig,
    TensorParallelConfig,
)
from dmg.tensor_parallel.rotary import GQAHeadExpander, RotaryEmbedding


class TestRotaryAndGQA(unittest.TestCase):
    """Test suite for RoPE embeddings and GQA head repeat logic."""

    def test_rope_frequencies_and_rotation(self):
        dim = 4
        rope = RotaryEmbedding(dim=dim, max_position_embeddings=128, base=10000.0)

        # Position 0 cos should be 1.0, sin should be 0.0
        self.assertEqual(rope.cos_cached[0], [1.0, 1.0])
        self.assertEqual(rope.sin_cached[0], [0.0, 0.0])

        # At position 0, applying rotary should be identity
        vec = [[1.0, 2.0, 3.0, 4.0]]
        rotated_0 = rope.apply_rotary(vec, start_pos=0)
        for i in range(4):
            self.assertAlmostEqual(rotated_0[0][i], vec[0][i], places=5)

    def test_rope_norm_preservation(self):
        # Rotation matrices are orthogonal: ||R(theta) x|| == ||x||
        dim = 4
        rope = RotaryEmbedding(dim=dim, max_position_embeddings=64)
        vec = [[3.0, 4.0, 1.0, 2.0]]
        orig_norm = math.sqrt(sum(x * x for x in vec[0]))

        rotated_pos5 = rope.apply_rotary(vec, start_pos=5)
        new_norm = math.sqrt(sum(x * x for x in rotated_pos5[0]))
        self.assertAlmostEqual(orig_norm, new_norm, places=5)

    def test_gqa_head_repeater(self):
        # 2 KV heads, head_dim = 2 -> total dim = 4
        # Repeat factor 2 -> 4 heads -> total dim = 8
        kv = [
            [1.0, 2.0, 10.0, 20.0],
            [3.0, 4.0, 30.0, 40.0],
        ]
        expanded = GQAHeadExpander.repeat_kv(
            kv, num_kv_heads=2, head_dim=2, repeat_factor=2
        )
        self.assertEqual(len(expanded), 2)
        self.assertEqual(len(expanded[0]), 8)
        # First row: head 0 repeated twice, head 1 repeated twice
        expected_row_0 = [1.0, 2.0, 1.0, 2.0, 10.0, 20.0, 10.0, 20.0]
        self.assertEqual(expanded[0], expected_row_0)

    def test_parallel_attention_with_rope_and_gqa(self):
        # World size = 2, total 4 Q heads, 2 KV heads -> 2 Q heads and 1 KV head per rank
        # Repeat factor = 2 Q heads / 1 KV head = 2
        # head_dim = 2 -> hidden_size = 8
        tp_cfg = TensorParallelConfig(world_size=2, rank=0)
        att_cfg = ParallelAttentionConfig(
            hidden_size=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            attention_type=AttentionType.GQA,
            use_rope=True,
            tp_config=tp_cfg,
        )

        # Projections for local rank:
        # Q: 4 rows (2 heads x 2 dim) x 8 cols
        # K: 2 rows (1 head x 2 dim) x 8 cols
        # V: 2 rows (1 head x 2 dim) x 8 cols
        # O: 8 rows x 4 cols (2 heads x 2 dim)
        w_q = [[0.5] * 8 for _ in range(4)]
        w_k = [[0.5] * 8 for _ in range(2)]
        w_v = [[0.5] * 8 for _ in range(2)]
        w_o = [[0.25] * 4 for _ in range(8)]

        q_proj = ColumnParallelLinear(w_q)
        k_proj = ColumnParallelLinear(w_k)
        v_proj = ColumnParallelLinear(w_v)
        o_proj = RowParallelLinear(w_o)

        attention = ParallelSelfAttention(att_cfg, q_proj, k_proj, v_proj, o_proj)

        # 2 tokens x 8 hidden dims
        x = [
            [1.0, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.8, 0.4, 0.3, 0.1, 0.0, 0.0, 0.0],
        ]
        out_local = attention.forward_local(x, causal_mask=True, start_pos=0)
        self.assertEqual(len(out_local), 2)
        self.assertEqual(len(out_local[0]), 8)


if __name__ == "__main__":
    unittest.main()
