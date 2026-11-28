"""
Parallel Rotary Position Embeddings (RoPE) and GQA Head Expansion.
Applies per-head rotary transformations locally on tensor parallel ranks
and aligns Grouped-Query Attention (GQA) key-value heads with query head groups.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

from dmg.tensor_parallel.exceptions import ParallelShapeMismatchError


class RotaryEmbedding:
    """
    Computes and caches Rotary Position Embeddings (RoPE) frequencies.
    Transformation is embarrassingly parallel across attention heads:
    each rank applies RoPE to its assigned local heads independently.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Inverse frequencies: theta_i = 1.0 / (base ** (2i / dim)) for i in [0, dim/2)
        half_dim = dim // 2
        self.inv_freq = [
            1.0 / (self.base ** ((2 * i) / dim)) for i in range(half_dim)
        ]

        # Precompute cos and sin tables: [max_pos, half_dim]
        self.cos_cached: List[List[float]] = []
        self.sin_cached: List[List[float]] = []
        for pos in range(max_position_embeddings):
            cos_row = [math.cos(pos * freq) for freq in self.inv_freq]
            sin_row = [math.sin(pos * freq) for freq in self.inv_freq]
            self.cos_cached.append(cos_row)
            self.sin_cached.append(sin_row)

    def apply_rotary(
        self,
        tensor: List[List[float]],
        start_pos: int = 0,
    ) -> List[List[float]]:
        """
        Apply 2D rotary embedding to tensor of shape [seq_len, head_dim].
        x = [x_1, x_2, ..., x_d] -> [-x_2, x_1, -x_4, x_3, ...]
        """
        seq_len = len(tensor)
        out = []
        half_dim = self.dim // 2

        for t in range(seq_len):
            pos = start_pos + t
            if pos >= self.max_position_embeddings:
                pos = pos % self.max_position_embeddings

            cos_r = self.cos_cached[pos]
            sin_r = self.sin_cached[pos]
            row = tensor[t]
            new_row = [0.0] * self.dim

            for i in range(half_dim):
                x1 = row[2 * i]
                x2 = row[2 * i + 1]
                c = cos_r[i]
                s = sin_r[i]
                # [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
                new_row[2 * i] = x1 * c - x2 * s
                new_row[2 * i + 1] = x1 * s + x2 * c

            out.append(new_row)

        return out

    def apply_rotary_to_heads(
        self,
        heads_tensor: List[List[float]],
        num_heads: int,
        head_dim: int,
        start_pos: int = 0,
    ) -> List[List[float]]:
        """
        Apply RoPE to multiple contiguous heads packed in [seq_len, num_heads * head_dim].
        """
        seq_len = len(heads_tensor)
        out = [[0.0] * (num_heads * head_dim) for _ in range(seq_len)]

        for h in range(num_heads):
            start = h * head_dim
            end = start + head_dim
            # Extract head h
            h_slice = [row[start:end] for row in heads_tensor]
            h_rotated = self.apply_rotary(h_slice, start_pos=start_pos)
            # Write back
            for t in range(seq_len):
                for d in range(head_dim):
                    out[t][start + d] = h_rotated[t][d]

        return out


class GQAHeadExpander:
    """
    Expands Key and Value heads for Grouped-Query Attention (GQA).
    If a rank has N_q query heads and N_kv key-value heads, each KV head
    is repeated (N_q // N_kv) times to match the query group size.
    """

    @staticmethod
    def repeat_kv(
        kv_tensor: List[List[float]],
        num_kv_heads: int,
        head_dim: int,
        repeat_factor: int,
    ) -> List[List[float]]:
        """
        Repeat each KV head `repeat_factor` times.
        Input: [seq_len, num_kv_heads * head_dim]
        Output: [seq_len, (num_kv_heads * repeat_factor) * head_dim]
        """
        if repeat_factor <= 1:
            return kv_tensor

        seq_len = len(kv_tensor)
        out = []

        for t in range(seq_len):
            row = []
            for h in range(num_kv_heads):
                start = h * head_dim
                end = start + head_dim
                head_vals = kv_tensor[t][start:end]
                for _ in range(repeat_factor):
                    row.extend(head_vals)
            out.append(row)

        return out
