"""
Output synchronization and end-to-end distributed attention module.
Combines local attention projections with cross-rank All-Reduce synchronization.
"""

from __future__ import annotations

import copy
from typing import Any, List, Optional, Tuple

from dmg.tensor_parallel.attention import ParallelSelfAttention
from dmg.tensor_parallel.config import ParallelAttentionConfig
from dmg.tensor_parallel.exceptions import CommunicationSyncError
from dmg.tensor_parallel.interfaces import CommunicatorProtocol


class AttentionOutputSynchronizer:
    """
    Synchronizes partial attention outputs across distributed worker ranks
    using All-Reduce (SUM) collective operations.
    """

    def __init__(
        self,
        communicator: CommunicatorProtocol,
        bias: Optional[Any] = None,
    ):
        self.communicator = communicator
        self.bias = bias

    def synchronize(
        self,
        local_output: Any,
        residual: Optional[Any] = None,
        op_id: Optional[str] = None,
    ) -> Any:
        """
        Synchronize local output projection across all ranks:
        Output = All-Reduce_SUM(local_output) + bias + residual
        """
        # Execute All-Reduce Sum
        if hasattr(self.communicator, "submit_tensor") and op_id:
            if local_output is not None:
                self.communicator.submit_tensor(op_id, local_output)
            reduced = self.communicator.all_reduce_sum(op_id=op_id)
        else:
            reduced = self.communicator.all_reduce_sum(local_output)

        # PyTorch bias & residual addition
        if hasattr(reduced, "__class__") and "Tensor" in reduced.__class__.__name__:
            if self.bias is not None:
                reduced = reduced + self.bias
            if residual is not None:
                reduced = reduced + residual
            return reduced

        # NumPy bias & residual addition
        if hasattr(reduced, "__class__") and "ndarray" in reduced.__class__.__name__:
            if self.bias is not None:
                reduced = reduced + self.bias
            if residual is not None:
                reduced = reduced + residual
            return reduced

        # Python lists fallback
        if isinstance(reduced, list):
            out = copy.deepcopy(reduced)
            seq_len = len(out)
            dim = len(out[0]) if seq_len > 0 else 0

            # Add bias
            if self.bias is not None:
                for i in range(seq_len):
                    for j in range(dim):
                        out[i][j] += self.bias[j]

            # Add residual
            if residual is not None:
                for i in range(seq_len):
                    for j in range(dim):
                        out[i][j] += residual[i][j]

            return out

        return reduced


class TensorParallelAttentionBlock:
    """
    High-level end-to-end attention block combining local attention computation
    with cross-rank collective output synchronization.
    """

    def __init__(
        self,
        attention: ParallelSelfAttention,
        synchronizer: AttentionOutputSynchronizer,
    ):
        self.attention = attention
        self.synchronizer = synchronizer

    def forward(
        self,
        hidden_states: Any,
        residual: Optional[Any] = None,
        causal_mask: bool = True,
        sync_op_id: Optional[str] = None,
    ) -> Any:
        """Execute complete forward pass with synchronized output."""
        # 1. Local head computation and row-parallel projection
        local_out = self.attention.forward_local(
            hidden_states, causal_mask=causal_mask
        )

        # 2. Cross-rank All-Reduce synchronization & residual addition
        sync_out = self.synchronizer.synchronize(
            local_out, residual=residual, op_id=sync_op_id
        )

        return sync_out
