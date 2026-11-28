"""
Speculative Decoding Parallel Attention Verification and Tree-Attention engine.
Accelerates distributed inference by concurrently verifying multiple draft candidate tokens
using custom Directed Acyclic Graph (DAG) attention masks and per-rank KV-cache rollback.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dmg.tensor_parallel.exceptions import TensorParallelError


@dataclass
class SpeculativeTreeMask:
    """
    Constructs and manages custom 2D/3D boolean attention masks for speculative token trees.
    In speculative decoding (Medusa, Eagle), candidate tokens are organized in a tree structure
    where each candidate can attend to its ancestors in the tree plus the prefix prompt tokens.
    """

    prefix_len: int
    tree_size: int
    parent_indices: List[int]  # Parent index for each token in tree (-1 for root candidates)
    # Shape: [tree_size, prefix_len + tree_size]
    mask: List[List[bool]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.mask:
            self.mask = self.build_mask()

    def build_mask(self) -> List[List[bool]]:
        """
        Build the tree attention mask.
        Each candidate token i attends to:
        1. All prefix tokens (0 to prefix_len - 1).
        2. Itself.
        3. All its ancestral parents up the tree.
        """
        total_k = self.prefix_len + self.tree_size
        mask = []

        for i in range(self.tree_size):
            row = [False] * total_k
            # 1. Attend to all prompt/prefix tokens
            for p in range(self.prefix_len):
                row[p] = True

            # 2. Attend to self
            row[self.prefix_len + i] = True

            # 3. Traverse parent chain
            curr_parent = self.parent_indices[i] if i < len(self.parent_indices) else -1
            while curr_parent >= 0:
                row[self.prefix_len + curr_parent] = True
                curr_parent = (
                    self.parent_indices[curr_parent]
                    if curr_parent < len(self.parent_indices)
                    else -1
                )

            mask.append(row)

        return mask


class SpeculativeAttentionVerifier:
    """
    Evaluates speculative candidate token attention across tensor parallel ranks.
    Executes branch verification, accepts matched tokens, and rolls back KV-caches.
    """

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.total_speculative_tokens_evaluated = 0
        self.total_tokens_accepted = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_speculative_tokens_evaluated == 0:
            return 0.0
        return self.total_tokens_accepted / self.total_speculative_tokens_evaluated

    @property
    def estimated_speedup(self) -> float:
        """Effective inference speedup factor: (1 + accepted_per_step)."""
        return 1.0 + self.acceptance_rate

    @staticmethod
    def evaluate_tree_attention(
        q_tree: List[List[float]],
        k_all: List[List[float]],
        v_all: List[List[float]],
        tree_mask: SpeculativeTreeMask,
    ) -> List[List[float]]:
        """
        Compute attention over speculative candidate tokens using the tree attention mask.
        q_tree: [tree_size, head_dim]
        k_all, v_all: [prefix_len + tree_size, head_dim]
        """
        tree_size = len(q_tree)
        total_k = len(k_all)
        head_dim = len(q_tree[0]) if tree_size > 0 else 0
        scale = 1.0 / math.sqrt(head_dim) if head_dim > 0 else 1.0

        out = []
        for i in range(tree_size):
            q_i = q_tree[i]
            mask_row = tree_mask.mask[i]

            # Compute dot products
            scores = []
            for j in range(total_k):
                if mask_row[j]:
                    dot = sum(q_i[d] * k_all[j][d] for d in range(head_dim)) * scale
                    scores.append(dot)
                else:
                    scores.append(-1e9)

            # Softmax
            max_s = max(scores)
            exps = [math.exp(s - max_s) if s > -1e8 else 0.0 for s in scores]
            sum_e = sum(exps)
            weights = [e / sum_e if sum_e > 0 else 0.0 for e in exps]

            # Output vector
            vec = [
                sum(weights[j] * v_all[j][d] for j in range(total_k))
                for d in range(head_dim)
            ]
            out.append(vec)

        return out

    def verify_and_prune_kv_cache(
        self,
        acceptance_flags: List[bool],
        candidate_k: List[List[float]],
        candidate_v: List[List[float]],
    ) -> Tuple[List[List[float]], List[List[float]], int]:
        """
        Filters candidate KV-cache representations based on verification results.
        Discards rejected branches and returns only accepted tokens to append to cache.
        """
        accepted_k = []
        accepted_v = []
        num_accepted = 0

        for i, accepted in enumerate(acceptance_flags):
            self.total_speculative_tokens_evaluated += 1
            if accepted and (i == 0 or acceptance_flags[i - 1]):
                # Sequential acceptance
                accepted_k.append(copy.deepcopy(candidate_k[i]))
                accepted_v.append(copy.deepcopy(candidate_v[i]))
                num_accepted += 1
                self.total_tokens_accepted += 1
            else:
                break

        return accepted_k, accepted_v, num_accepted
