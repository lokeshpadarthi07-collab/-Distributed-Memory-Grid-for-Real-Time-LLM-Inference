"""
Distributed Multi-Head Attention computation layers.
Implements ColumnParallelLinear, RowParallelLinear, and ParallelSelfAttention.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from dmg.tensor_parallel.config import ParallelAttentionConfig, TensorParallelConfig
from dmg.tensor_parallel.exceptions import ParallelShapeMismatchError
from dmg.tensor_parallel.interfaces import ParallelAttentionProtocol


def _matmul_2d(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Multiply 2D matrices A (M x K) and B (K x N)."""
    m = len(a)
    k = len(a[0]) if m > 0 else 0
    k2 = len(b)
    n = len(b[0]) if k2 > 0 else 0
    if k != k2:
        raise ParallelShapeMismatchError(f"Cannot multiply ({m}x{k}) and ({k2}x{n})")

    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for p in range(k):
            a_ip = a[i][p]
            if a_ip != 0.0:
                for j in range(n):
                    out[i][j] += a_ip * b[p][j]
    return out


def _transpose_2d(m: List[List[float]]) -> List[List[float]]:
    """Transpose a 2D matrix (M x N -> N x M)."""
    if not m:
        return []
    rows, cols = len(m), len(m[0])
    return [[m[r][c] for r in range(rows)] for c in range(cols)]


def _softmax_1d(row: List[float], mask_val: Optional[float] = None) -> List[float]:
    """Compute numerically stable Softmax over a 1D vector."""
    if not row:
        return []
    max_val = max(row)
    exp_vals = [math.exp(x - max_val) for x in row]
    sum_exp = sum(exp_vals)
    if sum_exp == 0.0:
        return [1.0 / len(row)] * len(row)
    return [v / sum_exp for v in exp_vals]


class ColumnParallelLinear:
    """
    Linear layer where the weight matrix is split along the output dimension (dim 0).
    Input: [..., in_features]
    Output: [..., out_features / world_size]
    """

    def __init__(
        self,
        weight_shard: Any,
        bias_shard: Optional[Any] = None,
    ):
        self.weight_shard = weight_shard
        self.bias_shard = bias_shard

    def forward(self, x: Any) -> Any:
        """Forward linear transformation: X @ W^T + b."""
        # PyTorch execution
        if hasattr(x, "__class__") and "Tensor" in x.__class__.__name__:
            import torch.nn.functional as F
            return F.linear(x, self.weight_shard, self.bias_shard)

        # NumPy execution
        if hasattr(x, "__class__") and "ndarray" in x.__class__.__name__:
            import numpy as np
            out = np.matmul(x, self.weight_shard.T)
            if self.bias_shard is not None:
                out = out + self.bias_shard
            return out

        # Pure Python fallback for list of vectors [seq_len, in_features]
        if isinstance(x, list) and isinstance(self.weight_shard, list):
            wt = _transpose_2d(self.weight_shard)
            out = _matmul_2d(x, wt)
            if self.bias_shard is not None:
                for row in out:
                    for j in range(len(row)):
                        row[j] += self.bias_shard[j]
            return out

        raise ParallelShapeMismatchError(f"Unsupported tensor type: {type(x)}")


class RowParallelLinear:
    """
    Linear layer where the weight matrix is split along the input dimension (dim 1).
    Input: [..., in_features / world_size]
    Output: Local partial sum of [..., out_features].
    Must be followed by All-Reduce Sum to complete the linear transformation.
    """

    def __init__(
        self,
        weight_shard: Any,
        bias: Optional[Any] = None,
    ):
        self.weight_shard = weight_shard
        self.bias = bias  # Bias is added after All-Reduce on rank 0 or all ranks

    def forward_local(self, x_local: Any) -> Any:
        """Compute localized partial matrix multiplication without communication."""
        if hasattr(x_local, "__class__") and "Tensor" in x_local.__class__.__name__:
            import torch
            return torch.matmul(x_local, self.weight_shard.t())

        if hasattr(x_local, "__class__") and "ndarray" in x_local.__class__.__name__:
            import numpy as np
            return np.matmul(x_local, self.weight_shard.T)

        if isinstance(x_local, list) and isinstance(self.weight_shard, list):
            wt = _transpose_2d(self.weight_shard)
            return _matmul_2d(x_local, wt)

        raise ParallelShapeMismatchError(f"Unsupported tensor type: {type(x_local)}")


class ParallelSelfAttention(ParallelAttentionProtocol):
    """
    Distributed Multi-Head Self-Attention layer.
    Each worker rank computes localized self-attention over its own assigned attention heads.
    """

    def __init__(
        self,
        config: ParallelAttentionConfig,
        q_proj: ColumnParallelLinear,
        k_proj: ColumnParallelLinear,
        v_proj: ColumnParallelLinear,
        o_proj: RowParallelLinear,
    ):
        self.config = config
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj

    def _compute_scaled_dot_product_python(
        self,
        q: List[List[float]],
        k: List[List[float]],
        v: List[List[float]],
        causal_mask: bool = True,
    ) -> List[List[float]]:
        """
        Pure Python implementation of Scaled Dot-Product Attention for a single head.
        q, k, v: [seq_len, head_dim]
        """
        seq_len = len(q)
        head_dim = len(q[0])
        scale = 1.0 / math.sqrt(head_dim)

        kt = _transpose_2d(k)
        scores = _matmul_2d(q, kt)  # [seq_len, seq_len]

        # Scale scores and apply causal masking
        for i in range(seq_len):
            for j in range(seq_len):
                scores[i][j] *= scale
                if causal_mask and j > i:
                    scores[i][j] = -1e9  # Mask future tokens

        # Softmax along columns
        attn_weights = [_softmax_1d(row) for row in scores]

        # Context: weights @ V -> [seq_len, head_dim]
        return _matmul_2d(attn_weights, v)

    def forward_local(
        self,
        hidden_states: Any,
        causal_mask: bool = True,
    ) -> Any:
        """
        Compute rank-local attention forward pass.
        Returns the un-synchronized local contribution to the output projection.
        """
        # Local Q, K, V projections
        q = self.q_proj.forward(hidden_states)
        k = self.k_proj.forward(hidden_states)
        v = self.v_proj.forward(hidden_states)

        head_dim = self.config.head_dim
        assert head_dim is not None
        num_local_heads = self.config.local_num_heads

        # Pure Python computation
        if isinstance(q, list):
            seq_len = len(q)
            # Separate heads
            local_head_outputs = []
            for h in range(num_local_heads):
                start = h * head_dim
                end = start + head_dim
                q_h = [row[start:end] for row in q]
                k_h = [row[start:end] for row in k]
                v_h = [row[start:end] for row in v]

                out_h = self._compute_scaled_dot_product_python(
                    q_h, k_h, v_h, causal_mask=causal_mask
                )
                local_head_outputs.append(out_h)

            # Concatenate local heads: [seq_len, num_local_heads * head_dim]
            local_attn_out = []
            for t in range(seq_len):
                row = []
                for h in range(num_local_heads):
                    row.extend(local_head_outputs[h][t])
                local_attn_out.append(row)

            # Local Row-Parallel Output Projection
            return self.o_proj.forward_local(local_attn_out)

        raise ParallelShapeMismatchError(f"Unsupported hidden_states type: {type(hidden_states)}")

    def forward(
        self,
        hidden_states: Any,
        attention_mask: Optional[Any] = None,
        kv_cache: Optional[Any] = None,
    ) -> Tuple[Any, Optional[Any]]:
        """Implementation of ParallelAttentionProtocol contract."""
        out = self.forward_local(hidden_states)
        return out, kv_cache
