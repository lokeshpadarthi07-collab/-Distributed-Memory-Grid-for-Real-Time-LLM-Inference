"""
Unit tests for PagedKVCacheManager, PhysicalBlock, and BlockTable.
"""

import unittest

from dmg.tensor_parallel.exceptions import TensorParallelError
from dmg.tensor_parallel.paged_cache import (
    BlockTable,
    PagedKVCacheManager,
    PhysicalBlock,
)


class TestPagedKVCache(unittest.TestCase):
    """Test suite for PagedAttention memory management."""

    def test_block_allocation_and_paging(self):
        # 4 total blocks, 4 tokens per block, 1 head x 2 dims
        mgr = PagedKVCacheManager(
            num_blocks=4,
            block_size=4,
            num_heads=1,
            head_dim=2,
        )

        self.assertEqual(mgr.num_free_blocks, 4)
        self.assertEqual(mgr.memory_utilization_ratio, 0.0)

        # Append 6 tokens (should spill over 1 block into 2 blocks: 4 + 2)
        k_tokens = [[float(i), float(i)] for i in range(6)]
        v_tokens = [[float(i) * 0.1, float(i) * 0.1] for i in range(6)]

        mgr.append_tokens("req_1", layer_idx=0, new_k=k_tokens, new_v=v_tokens)

        # 2 blocks should be allocated
        self.assertEqual(mgr.num_free_blocks, 2)
        self.assertEqual(mgr.memory_utilization_ratio, 0.5)

        table = mgr.get_or_create_table("req_1", layer_idx=0)
        self.assertEqual(len(table.physical_block_ids), 2)
        self.assertEqual(table.total_tokens, 6)

        # Read back contiguous sequence and verify
        k_read, v_read = mgr.read_contiguous_sequence("req_1", layer_idx=0)
        self.assertEqual(len(k_read), 6)
        self.assertEqual(len(v_read), 6)
        self.assertEqual(k_read, k_tokens)
        self.assertEqual(v_read, v_tokens)

    def test_free_and_block_recycling(self):
        mgr = PagedKVCacheManager(
            num_blocks=2,
            block_size=2,
            num_heads=1,
            head_dim=2,
        )

        # Use up all 2 blocks (4 tokens)
        k = [[1.0, 1.0]] * 4
        v = [[0.5, 0.5]] * 4
        mgr.append_tokens("req_temp", layer_idx=0, new_k=k, new_v=v)
        self.assertEqual(mgr.num_free_blocks, 0)

        # Free request
        mgr.free_request("req_temp")
        self.assertEqual(mgr.num_free_blocks, 2)
        self.assertEqual(mgr.memory_utilization_ratio, 0.0)

    def test_out_of_blocks_error(self):
        mgr = PagedKVCacheManager(
            num_blocks=1,
            block_size=2,
            num_heads=1,
            head_dim=2,
        )

        # 3 tokens exceeds 1 block of size 2
        k = [[1.0, 1.0]] * 3
        v = [[0.5, 0.5]] * 3
        with self.assertRaises(TensorParallelError):
            mgr.append_tokens("req_overflow", layer_idx=0, new_k=k, new_v=v)


if __name__ == "__main__":
    unittest.main()
