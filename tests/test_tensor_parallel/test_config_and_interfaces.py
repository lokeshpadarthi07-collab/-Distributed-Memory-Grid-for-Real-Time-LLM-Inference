"""
Tests for TensorParallelConfig and ParallelAttentionConfig validation.
"""

import unittest

from dmg.tensor_parallel.config import (
    AttentionType,
    CommBackend,
    ParallelAttentionConfig,
    TensorParallelConfig,
)
from dmg.tensor_parallel.exceptions import InvalidParallelConfigError


class TestTensorParallelConfig(unittest.TestCase):
    """Test suite for TensorParallelConfig dataclass."""

    def test_valid_config(self):
        cfg = TensorParallelConfig(world_size=4, rank=2, backend=CommBackend.MOCK)
        self.assertEqual(cfg.world_size, 4)
        self.assertEqual(cfg.rank, 2)
        self.assertEqual(cfg.backend, CommBackend.MOCK)

    def test_invalid_world_size(self):
        with self.assertRaises(InvalidParallelConfigError):
            TensorParallelConfig(world_size=0, rank=0)
        with self.assertRaises(InvalidParallelConfigError):
            TensorParallelConfig(world_size=-2, rank=0)

    def test_invalid_rank(self):
        with self.assertRaises(InvalidParallelConfigError):
            TensorParallelConfig(world_size=4, rank=4)
        with self.assertRaises(InvalidParallelConfigError):
            TensorParallelConfig(world_size=4, rank=-1)


class TestParallelAttentionConfig(unittest.TestCase):
    """Test suite for ParallelAttentionConfig dataclass."""

    def test_valid_mha_config(self):
        tp_cfg = TensorParallelConfig(world_size=4, rank=1)
        att_cfg = ParallelAttentionConfig(
            hidden_size=4096,
            num_attention_heads=32,
            tp_config=tp_cfg,
        )
        self.assertEqual(att_cfg.head_dim, 128)
        self.assertEqual(att_cfg.local_num_heads, 8)
        self.assertEqual(att_cfg.local_num_kv_heads, 8)
        self.assertEqual(att_cfg.local_hidden_size, 1024)

    def test_gqa_config(self):
        tp_cfg = TensorParallelConfig(world_size=4, rank=0)
        att_cfg = ParallelAttentionConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            attention_type=AttentionType.GQA,
            tp_config=tp_cfg,
        )
        self.assertEqual(att_cfg.local_num_heads, 8)
        self.assertEqual(att_cfg.local_num_kv_heads, 2)

    def test_indivisible_heads(self):
        tp_cfg = TensorParallelConfig(world_size=6, rank=0)
        with self.assertRaises(InvalidParallelConfigError):
            ParallelAttentionConfig(
                hidden_size=4096,
                num_attention_heads=32,  # 32 not divisible by 6
                tp_config=tp_cfg,
            )

    def test_invalid_hidden_size(self):
        with self.assertRaises(InvalidParallelConfigError):
            ParallelAttentionConfig(hidden_size=-100, num_attention_heads=16)


if __name__ == "__main__":
    unittest.main()
