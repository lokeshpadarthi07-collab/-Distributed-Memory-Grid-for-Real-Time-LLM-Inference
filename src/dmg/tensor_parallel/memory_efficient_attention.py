"""
Memory-efficient chunked attention kernel and online softmax engine.
Implements tiled scaled dot-product attention (FlashAttention-style online softmax)
to prevent O(seq_len^2) memory bottlenecks during long-context distributed inference.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

from dmg.tensor_parallel.exceptions import ParallelShapeMismatchError


class OnlineSoftmaxAccumulator:
    """
    Maintains running online softmax statistics (max and sum of exponentials)
    enabling tiled attention computation without materializing the full N x N matrix.
    """

    def __init__(self, head_dim: int):
        self.head_dim = head_dim
        # Running max m_i per query token (initialized to -infinity)
        self.m: float = -float("inf")
        # Running denominator d_i per query token
        self.d: float = 0.0
        # Running unnormalized output vector: [head_dim]
        self.acc: List[float] = [0.0] * head_dim

    def update(
        self,
        scores_chunk: List[float],
        v_chunk: List[List[float]],
    ) -> None:
        """
        Update accumulator with a new column tile of attention scores and V vectors.
        scores_chunk: [tile_len]
        v_chunk: [tile_len, head_dim]
        """
        if not scores_chunk:
            return

        tile_max = max(scores_chunk)
        if tile_max == -float("inf"):
            return

        # New global maximum between previous tiles and current tile
        m_new = max(self.m, tile_max)

        # Scale factor for previous accumulated denominator and output
        alpha = math.exp(self.m - m_new) if self.m != -float("inf") else 0.0

        # Compute exp(score - m_new) for current tile
        p_tile = [math.exp(s - m_new) if s != -float("inf") else 0.0 for s in scores_chunk]
        d_tile = sum(p_tile)

        # Update denominator
        self.d = alpha * self.d + d_tile
        self.m = m_new

        # Update numerator accumulator: alpha * acc + p_tile @ v_chunk
        tile_len = len(scores_chunk)
        for j in range(self.head_dim):
            p_dot_v = sum(p_tile[k] * v_chunk[k][j] for k in range(tile_len))
            self.acc[j] = alpha * self.acc[j] + p_dot_v

    def finalize(self) -> List[float]:
        """Normalize accumulated output by total denominator d."""
        if self.d == 0.0:
            return [0.0] * self.head_dim
        return [val / self.d for val in self.acc]


class ChunkedAttentionKernel:
    """
    Tiled Scaled Dot-Product Attention kernel.
    Evaluates attention over query and key/value chunks, keeping intermediate
    activation memory bounded by the block tile size rather than sequence length.
    """

    @staticmethod
    def chunked_scaled_dot_product(
        q: List[List[float]],
        k: List[List[float]],
        v: List[List[float]],
        block_size: int = 64,
        causal_mask: bool = True,
        sliding_window: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Compute attention(Q, K, V) in tiles of key/values.
        q: [seq_q, head_dim]
        k: [seq_k, head_dim]
        v: [seq_k, head_dim]
        """
        seq_q = len(q)
        seq_k = len(k)
        if seq_q == 0 or seq_k == 0:
            return []

        head_dim = len(q[0])
        scale = 1.0 / math.sqrt(head_dim)
        out = []

        # Process each query token with an independent online softmax accumulator
        for i in range(seq_q):
            q_i = q[i]
            accum = OnlineSoftmaxAccumulator(head_dim=head_dim)

            # Tile over keys/values in chunks of block_size
            for k_start in range(0, seq_k, block_size):
                k_end = min(k_start + block_size, seq_k)
                k_chunk = k[k_start:k_end]
                v_chunk = v[k_start:k_end]

                # Compute dot-product scores for tile: [tile_len]
                scores_tile = []
                for idx, k_vec in enumerate(k_chunk):
                    col_idx = k_start + idx

                    # Check causal masking
                    if causal_mask and col_idx > i:
                        scores_tile.append(-float("inf"))
                        continue

                    # Check sliding window attention
                    if sliding_window is not None and (i - col_idx) > sliding_window:
                        scores_tile.append(-float("inf"))
                        continue

                    dot = sum(q_i[d] * k_vec[d] for d in range(head_dim)) * scale
                    scores_tile.append(dot)

                accum.update(scores_tile, v_chunk)

            out.append(accum.finalize())

        return out
