"""
Unit tests for SpeculativeTreeMask and SpeculativeAttentionVerifier.
Verifies tree-attention DAG mask building, candidate token attention, and KV-cache pruning.
"""

import unittest

from dmg.tensor_parallel.speculative import (
    SpeculativeAttentionVerifier,
    SpeculativeTreeMask,
)


class TestSpeculativeAttention(unittest.TestCase):
    """Test suite for speculative decoding tree attention and verification."""

    def test_tree_mask_dag_construction(self):
        # 4 prompt tokens, 3 candidate draft tokens in tree:
        # Candidate 0: child of prompt (parent = -1)
        # Candidate 1: child of 0 (parent = 0)
        # Candidate 2: alternative branch, child of prompt (parent = -1)
        tree_mask = SpeculativeTreeMask(
            prefix_len=4,
            tree_size=3,
            parent_indices=[-1, 0, -1],
        )

        mask = tree_mask.mask
        self.assertEqual(len(mask), 3)
        self.assertEqual(len(mask[0]), 7)  # 4 prefix + 3 candidates

        # Candidate 0 attends to prefix [0..3] and self [4], but NOT [5] or [6]
        self.assertEqual(mask[0], [True, True, True, True, True, False, False])

        # Candidate 1 attends to prefix [0..3], parent [4], and self [5], but NOT [6]
        self.assertEqual(mask[1], [True, True, True, True, True, True, False])

        # Candidate 2 attends to prefix [0..3] and self [6], but NOT [4] or [5]
        self.assertEqual(mask[2], [True, True, True, True, False, False, True])

    def test_evaluate_tree_attention(self):
        # Tree size = 2, prefix = 2, head_dim = 2
        tree_mask = SpeculativeTreeMask(prefix_len=2, tree_size=2, parent_indices=[-1, 0])

        q_tree = [[1.0, 0.0], [0.0, 1.0]]
        k_all = [
            [1.0, 0.0],  # prefix 0
            [0.0, 1.0],  # prefix 1
            [1.0, 0.0],  # cand 0
            [0.0, 1.0],  # cand 1
        ]
        v_all = [
            [2.0, 0.0],
            [0.0, 2.0],
            [4.0, 0.0],
            [0.0, 4.0],
        ]

        out = SpeculativeAttentionVerifier.evaluate_tree_attention(
            q_tree=q_tree,
            k_all=k_all,
            v_all=v_all,
            tree_mask=tree_mask,
        )

        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0]), 2)
        # Cand 0 attends only to prefix + self (indices 0, 1, 2)
        self.assertGreater(out[0][0], 0.0)

    def test_verify_and_prune_kv_cache(self):
        verifier = SpeculativeAttentionVerifier(rank=0, world_size=2)

        candidates_k = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        candidates_v = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]

        # Candidate 0 accepted, candidate 1 accepted, candidate 2 rejected
        flags = [True, True, False]
        acc_k, acc_v, count = verifier.verify_and_prune_kv_cache(
            acceptance_flags=flags,
            candidate_k=candidates_k,
            candidate_v=candidates_v,
        )

        self.assertEqual(count, 2)
        self.assertEqual(len(acc_k), 2)
        self.assertEqual(len(acc_v), 2)
        self.assertEqual(acc_k[0], [1.0, 1.0])
        self.assertEqual(acc_k[1], [2.0, 2.0])
        self.assertEqual(verifier.total_tokens_accepted, 2)
        self.assertGreater(verifier.estimated_speedup, 1.0)


if __name__ == "__main__":
    unittest.main()
