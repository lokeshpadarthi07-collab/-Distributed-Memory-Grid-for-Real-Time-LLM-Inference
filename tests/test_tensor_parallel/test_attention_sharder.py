"""
Unit tests for AttentionHeadSharder and AttentionParallelismProvider protocol implementation.
Tests MHA, GQA, and MQA partition generation and weight sharding.
"""

import unittest

from dmg.core.interfaces import AttentionParallelismProvider
from dmg.tensor_parallel.config import (
    AttentionType,
    ParallelAttentionConfig,
    TensorParallelConfig,
)
from dmg.tensor_parallel.sharding import AttentionHeadSharder
from tests.test_tensor_parallel.test_tensor_splitting import Dummy2DTensor


class TestAttentionHeadSharder(unittest.TestCase):
    """Test suite for attention head sharding across ranks."""

    def test_mha_head_partition(self):
        # 32 heads, 4 ranks -> 8 heads per rank
        tp_cfg = TensorParallelConfig(world_size=4, rank=2)
        att_cfg = ParallelAttentionConfig(
            hidden_size=4096,
            num_attention_heads=32,
            tp_config=tp_cfg,
        )
        sharder = AttentionHeadSharder(att_cfg)

        part = sharder.compute_head_partition(rank=2, world_size=4)
        self.assertEqual(part.query_heads_range, (16, 24))
        self.assertEqual(part.kv_heads_range, (16, 24))
        self.assertEqual(part.num_query_heads, 8)
        self.assertEqual(part.num_kv_heads, 8)
        self.assertFalse(part.is_kv_replicated)

    def test_gqa_head_partition(self):
        # 32 query heads, 8 KV heads, 4 ranks
        # Query: 8 per rank. KV: 8 / 4 = 2 per rank.
        tp_cfg = TensorParallelConfig(world_size=4, rank=1)
        att_cfg = ParallelAttentionConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            attention_type=AttentionType.GQA,
            tp_config=tp_cfg,
        )
        sharder = AttentionHeadSharder(att_cfg)
        part = sharder.compute_head_partition(rank=1, world_size=4)
        self.assertEqual(part.query_heads_range, (8, 16))
        self.assertEqual(part.kv_heads_range, (2, 4))
        self.assertEqual(part.num_query_heads, 8)
        self.assertEqual(part.num_kv_heads, 2)
        self.assertFalse(part.is_kv_replicated)

    def test_mqa_head_partition_replicated_kv(self):
        # 32 query heads, 1 KV head, 4 ranks
        # Query: 8 per rank. KV: 1 head replicated across all ranks.
        tp_cfg = TensorParallelConfig(world_size=4, rank=3)
        att_cfg = ParallelAttentionConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=1,
            attention_type=AttentionType.MQA,
            tp_config=tp_cfg,
        )
        sharder = AttentionHeadSharder(att_cfg)
        part = sharder.compute_head_partition(rank=3, world_size=4)
        self.assertEqual(part.query_heads_range, (24, 32))
        self.assertEqual(part.kv_heads_range, (0, 1))
        self.assertTrue(part.is_kv_replicated)

    def test_protocol_implementation(self):
        tp_cfg = TensorParallelConfig(world_size=2, rank=0)
        att_cfg = ParallelAttentionConfig(
            hidden_size=2048,
            num_attention_heads=16,
            tp_config=tp_cfg,
        )
        sharder = AttentionHeadSharder(att_cfg)
        # Check runtime Protocol compatibility with Member 2 interface
        self.assertTrue(isinstance(sharder, AttentionParallelismProvider))

        spec = sharder.get_attention_shard_spec(layer_idx=5, rank=1, world_size=2)
        self.assertEqual(spec["layer_idx"], 5)
        self.assertEqual(spec["rank"], 1)
        self.assertEqual(spec["num_query_heads"], 8)
        self.assertEqual(spec["query_heads_start"], 8)
        self.assertEqual(spec["query_heads_end"], 16)

    def test_shard_layer_weights(self):
        # 4 heads, head_dim = 2 -> total hidden = 8
        tp_cfg = TensorParallelConfig(world_size=2, rank=0)
        att_cfg = ParallelAttentionConfig(
            hidden_size=8,
            num_attention_heads=4,
            head_dim=2,
            tp_config=tp_cfg,
        )
        sharder = AttentionHeadSharder(att_cfg)

        # 8 x 8 weight matrix
        w = Dummy2DTensor([[i * 8 + j for j in range(8)] for i in range(8)])
        manifest = sharder.shard_layer_weights(
            layer_idx=0,
            rank=0,
            world_size=2,
            q_weight=w,
            k_weight=w,
            v_weight=w,
            o_weight=w,
        )

        # Rank 0 gets first 2 heads of Q, K, V -> 4 rows x 8 cols
        self.assertEqual(manifest.q_weight_shard.shape, (4, 8))
        self.assertEqual(manifest.k_weight_shard.shape, (4, 8))
        self.assertEqual(manifest.v_weight_shard.shape, (4, 8))
        # O weight is row-parallel: 8 rows x 4 cols
        self.assertEqual(manifest.o_weight_shard.shape, (8, 4))


if __name__ == "__main__":
    unittest.main()
