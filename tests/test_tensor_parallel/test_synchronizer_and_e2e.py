"""
Unit and integration tests for AttentionOutputSynchronizer and TensorParallelAttentionBlock.
Tests full multi-rank distributed self-attention with All-Reduce synchronization.
"""

import unittest

from dmg.tensor_parallel.attention import (
    ColumnParallelLinear,
    ParallelSelfAttention,
    RowParallelLinear,
)
from dmg.tensor_parallel.communicator import MockCommunicator, MockCommunicatorGroup
from dmg.tensor_parallel.config import (
    CommBackend,
    ParallelAttentionConfig,
    TensorParallelConfig,
)
from dmg.tensor_parallel.sharding import AttentionHeadSharder
from dmg.tensor_parallel.synchronizer import (
    AttentionOutputSynchronizer,
    TensorParallelAttentionBlock,
)


class TestSynchronizerAndE2E(unittest.TestCase):
    """Test suite for end-to-end tensor parallel attention with output synchronization."""

    def test_output_synchronizer(self):
        # 2 ranks
        group = MockCommunicatorGroup(world_size=2)
        comm0 = MockCommunicator(
            TensorParallelConfig(world_size=2, rank=0), shared_group=group
        )
        comm1 = MockCommunicator(
            TensorParallelConfig(world_size=2, rank=1), shared_group=group
        )

        sync0 = AttentionOutputSynchronizer(comm0, bias=[1.0, 1.0])
        sync1 = AttentionOutputSynchronizer(comm1, bias=[1.0, 1.0])

        out0_local = [[2.0, 3.0]]
        out1_local = [[4.0, 5.0]]
        residual = [[10.0, 10.0]]

        comm0.submit_tensor("sync_step", out0_local)
        comm1.submit_tensor("sync_step", out1_local)

        res0 = sync0.synchronize(out0_local, residual=residual, op_id="sync_step")
        res1 = sync1.synchronize(out1_local, residual=residual, op_id="sync_step")

        # Reduced sum = [2+4, 3+5] = [6.0, 8.0]
        # + bias [1.0, 1.0] = [7.0, 9.0]
        # + residual [10.0, 10.0] = [17.0, 19.0]
        expected = [[17.0, 19.0]]
        self.assertEqual(res0, expected)
        self.assertEqual(res1, expected)

    def test_end_to_end_parallel_attention_pipeline(self):
        # Total 2 heads, world_size = 2 -> 1 head per rank. head_dim = 2 -> hidden_size = 4
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

            # Local weights for head r
            w_q = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
            w_k = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
            w_v = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
            w_o = [[0.5, 0.0], [0.0, 0.5], [0.5, 0.0], [0.0, 0.5]]

            q_p = ColumnParallelLinear(w_q)
            k_p = ColumnParallelLinear(w_k)
            v_p = ColumnParallelLinear(w_v)
            o_p = RowParallelLinear(w_o)

            attn = ParallelSelfAttention(att_cfg, q_p, k_p, v_p, o_p)
            sync = AttentionOutputSynchronizer(comm, bias=[0.1, 0.1, 0.1, 0.1])
            block = TensorParallelAttentionBlock(attn, sync)
            blocks.append(block)

        # Common input
        x = [
            [1.0, 0.5, 0.2, 0.1],
            [0.2, 0.8, 0.4, 0.3],
        ]

        # Step 1: Each rank computes local attention and submits
        for r in range(world_size):
            local_out = blocks[r].attention.forward_local(x)
            blocks[r].synchronizer.communicator.submit_tensor("e2e_sync", local_out)

        # Step 2: All ranks synchronize
        results = [
            blocks[r].synchronizer.synchronize(op_id="e2e_sync", local_output=None)
            for r in range(world_size)
        ]

        # All ranks must achieve identical synchronized output
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(results[0]), 2)     # 2 tokens
        self.assertEqual(len(results[0][0]), 4)  # 4 hidden dims


if __name__ == "__main__":
    unittest.main()
