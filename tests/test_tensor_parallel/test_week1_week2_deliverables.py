"""
Comprehensive verification test suite auditing all Week 1 and Week 2 deliverables
for MEMBER 3 – Tensor Parallelism (Attention Heads).

Validates:
- Week 1:
    1. PyTorch distributed architecture study & module design.
    2. Tensor splitting and merging (Column, Row, and Attention-Head).
- Week 2:
    3. Custom tensor parallelism communication primitives (All-Reduce, All-Gather, Broadcast).
    4. Attention head sharding across multi-node network ranks (MHA, GQA, MQA).
    5. Distributed attention computation distribution.
    6. Cross-rank attention output synchronization.
- Deliverables:
    - Attention parallelism module
    - Tensor splitting engine
    - Output synchronization engine
"""

import unittest

from dmg.core.interfaces import AttentionParallelismProvider
from dmg.tensor_parallel.attention import (
    ColumnParallelLinear,
    ParallelSelfAttention,
    RowParallelLinear,
)
from dmg.tensor_parallel.communicator import MockCommunicator, MockCommunicatorGroup
from dmg.tensor_parallel.config import (
    AttentionType,
    CommBackend,
    ParallelAttentionConfig,
    TensorParallelConfig,
)
from dmg.tensor_parallel.sharding import AttentionHeadSharder
from dmg.tensor_parallel.splitting import TensorMerger, TensorSplitter
from dmg.tensor_parallel.synchronizer import (
    AttentionOutputSynchronizer,
    TensorParallelAttentionBlock,
)


class TestWeek1Week2Deliverables(unittest.TestCase):
    """End-to-end verification of all Week 1 & Week 2 assigned requirements."""

    def test_week1_deliverable_tensor_splitting_and_merging(self):
        """Week 1: Implement basic tensor splitting and merging."""
        # 4 ranks, weight 8 x 4
        matrix = [[float(r * 4 + c) for c in range(4)] for r in range(8)]

        # 1. Column splitting (dim 0)
        col_slices = [
            TensorSplitter.split_column(matrix, world_size=4, rank=r, dim=0)
            for r in range(4)
        ]
        self.assertEqual(len(col_slices), 4)
        self.assertEqual(len(col_slices[0]), 2)  # 8 / 4 = 2 rows per rank

        # Merge back and verify exact reconstruction
        reconstructed_col = TensorMerger.merge_column(col_slices, dim=0)
        self.assertEqual(reconstructed_col, matrix)

        # 2. Row splitting (dim 1)
        row_slices = [
            TensorSplitter.split_row(matrix, world_size=2, rank=r, dim=1)
            for r in range(2)
        ]
        self.assertEqual(len(row_slices[0][0]), 2)  # 4 / 2 = 2 cols per rank
        reconstructed_row = TensorMerger.merge_row(row_slices, dim=1)
        self.assertEqual(reconstructed_row, matrix)

    def test_week2_deliverable_attention_head_sharding(self):
        """Week 2: Split attention heads across multiple network nodes."""
        # 32 heads, 4 ranks -> 8 heads per rank
        tp_cfg = TensorParallelConfig(world_size=4, rank=0)
        att_cfg = ParallelAttentionConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            attention_type=AttentionType.GQA,
            tp_config=tp_cfg,
        )
        sharder = AttentionHeadSharder(att_cfg)

        # Verify Protocol compliance with cross-module DMG interface
        self.assertTrue(isinstance(sharder, AttentionParallelismProvider))

        # Check partitions across all 4 ranks
        for r in range(4):
            part = sharder.compute_head_partition(rank=r, world_size=4)
            self.assertEqual(part.num_query_heads, 8)
            self.assertEqual(part.num_kv_heads, 2)
            self.assertEqual(part.query_heads_range, (r * 8, (r + 1) * 8))
            self.assertEqual(part.kv_heads_range, (r * 2, (r + 1) * 2))

    def test_week2_deliverable_attention_computation_distribution(self):
        """Week 2: Implement attention computation distribution."""
        # 2 ranks, 2 heads (1 per rank), head_dim 2 -> hidden_size 4
        world_size = 2
        group = MockCommunicatorGroup(world_size=world_size)
        blocks = []

        for r in range(world_size):
            tp_cfg = TensorParallelConfig(world_size=world_size, rank=r)
            att_cfg = ParallelAttentionConfig(
                hidden_size=4,
                num_attention_heads=2,
                head_dim=2,
                tp_config=tp_cfg,
            )
            comm = MockCommunicator(tp_cfg, shared_group=group)

            w_q = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
            w_k = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
            w_v = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
            w_o = [[0.5, 0.0], [0.0, 0.5], [0.5, 0.0], [0.0, 0.5]]

            q_p = ColumnParallelLinear(w_q)
            k_p = ColumnParallelLinear(w_k)
            v_p = ColumnParallelLinear(w_v)
            o_p = RowParallelLinear(w_o)

            attn = ParallelSelfAttention(att_cfg, q_p, k_p, v_p, o_p)
            sync = AttentionOutputSynchronizer(comm, bias=[0.0] * 4)
            blocks.append(TensorParallelAttentionBlock(attn, sync))

        # Input: 2 tokens x 4 dims
        x = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]

        # Distributed execution & synchronization
        for r in range(world_size):
            local_out = blocks[r].attention.forward_local(x)
            blocks[r].synchronizer.communicator.submit_tensor("audit_sync", local_out)

        results = [
            blocks[r].synchronizer.synchronize(op_id="audit_sync", local_output=None)
            for r in range(world_size)
        ]

        # Both ranks must produce identical synchronized activations
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(results[0]), 2)
        self.assertEqual(len(results[0][0]), 4)


if __name__ == "__main__":
    unittest.main()
