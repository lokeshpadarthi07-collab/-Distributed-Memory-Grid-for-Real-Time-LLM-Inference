"""
Unit tests for DistributedKVCacheManager and TensorParallelProfiler.
"""

import unittest

from dmg.tensor_parallel.config import ParallelAttentionConfig, TensorParallelConfig
from dmg.tensor_parallel.exceptions import TensorParallelError
from dmg.tensor_parallel.kv_cache import DistributedKVCacheManager
from dmg.tensor_parallel.profiler import TensorParallelProfiler


class TestKVCacheAndProfiler(unittest.TestCase):
    """Test suite for KV-Cache management and latency profiling."""

    def test_kv_cache_lifecycle(self):
        tp_cfg = TensorParallelConfig(world_size=2, rank=0)
        att_cfg = ParallelAttentionConfig(
            hidden_size=8,
            num_attention_heads=4,
            head_dim=2,
            tp_config=tp_cfg,
        )
        mgr = DistributedKVCacheManager(att_cfg, max_seq_len=10)

        # Prefill: 2 tokens
        k0 = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        v0 = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        k_cached, v_cached = mgr.update_cache(layer_idx=0, new_k=k0, new_v=v0)

        self.assertEqual(len(k_cached), 2)
        self.assertEqual(len(v_cached), 2)

        # Incremental token append: 1 token
        k1 = [[9.0, 10.0, 11.0, 12.0]]
        v1 = [[0.9, 1.0, 1.1, 1.2]]
        k_cached2, v_cached2 = mgr.update_cache(layer_idx=0, new_k=k1, new_v=v1)

        self.assertEqual(len(k_cached2), 3)
        self.assertEqual(len(v_cached2), 3)
        self.assertEqual(k_cached2[2], k1[0])

        # Memory calculation
        self.assertGreater(mgr.get_memory_bytes(), 0)

        # Clear
        mgr.clear(layer_idx=0)
        state = mgr.get_or_create(layer_idx=0)
        self.assertTrue(state.is_empty)

    def test_kv_cache_overflow(self):
        tp_cfg = TensorParallelConfig(world_size=1, rank=0)
        att_cfg = ParallelAttentionConfig(hidden_size=4, num_attention_heads=2, tp_config=tp_cfg)
        mgr = DistributedKVCacheManager(att_cfg, max_seq_len=2)

        k = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        v = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        with self.assertRaises(TensorParallelError):
            mgr.update_cache(layer_idx=0, new_k=k, new_v=v)

    def test_profiler_metrics(self):
        prof = TensorParallelProfiler(rank=0, world_size=2)
        prof.record_step(
            step_id="step_1",
            qkv_ms=2.5,
            attn_ms=3.0,
            out_ms=1.5,
            sync_ms=1.0,
        )
        summary = prof.get_summary()
        self.assertEqual(summary["total_steps"], 1)
        self.assertEqual(summary["avg_compute_ms"], 7.0)
        self.assertEqual(summary["avg_sync_ms"], 1.0)
        self.assertAlmostEqual(summary["avg_total_ms"], 8.0)
        self.assertGreater(summary["comm_overhead_percent"], 0)


if __name__ == "__main__":
    unittest.main()
