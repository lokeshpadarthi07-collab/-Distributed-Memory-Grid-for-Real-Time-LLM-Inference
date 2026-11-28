"""
Tensor splitting and merging engine for Tensor Parallelism.
Supports PyTorch Tensors, NumPy ndarrays, and native Python numeric structures.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from dmg.tensor_parallel.exceptions import (
    AttentionHeadSplitError,
    HeadReconstructionError,
    ParallelShapeMismatchError,
)


def _get_shape(tensor: Any) -> Tuple[int, ...]:
    """Retrieve tensor shape across PyTorch, NumPy, or nested lists."""
    if hasattr(tensor, "shape"):
        return tuple(tensor.shape)
    if isinstance(tensor, list):
        shape = []
        curr = tensor
        while isinstance(curr, list):
            shape.append(len(curr))
            curr = curr[0] if len(curr) > 0 else 0
        return tuple(shape)
    raise ParallelShapeMismatchError(f"Unsupported tensor type: {type(tensor)}")


def _slice_dim(tensor: Any, dim: int, start: int, end: int) -> Any:
    """Slice tensor along specified dimension across PyTorch, NumPy, and lists."""
    if isinstance(tensor, list):
        if dim == 0:
            return tensor[start:end]
        elif dim == 1:
            return [row[start:end] for row in tensor]
        raise NotImplementedError(f"List slicing along dim {dim} not supported.")

    if hasattr(tensor, "__getitem__"):
        ndim = len(_get_shape(tensor))
        dim = dim if dim >= 0 else ndim + dim
        idx = [slice(None)] * ndim
        idx[dim] = slice(start, end)
        return tensor[tuple(idx)]
    raise ParallelShapeMismatchError(f"Cannot slice object of type {type(tensor)}")


def _concat_dim(tensors: Sequence[Any], dim: int) -> Any:
    """Concatenate sequence of tensors along specified dimension."""
    if not tensors:
        raise HeadReconstructionError("Cannot concatenate empty list of tensors.")

    first = tensors[0]
    # Check for PyTorch tensor
    if hasattr(first, "cat") or (hasattr(first, "__class__") and "Tensor" in first.__class__.__name__):
        try:
            import torch
            return torch.cat(list(tensors), dim=dim)
        except ImportError:
            pass

    # Check for NumPy array
    if hasattr(first, "__class__") and "ndarray" in first.__class__.__name__:
        try:
            import numpy as np
            return np.concatenate(list(tensors), axis=dim)
        except ImportError:
            pass

    # Fallback for Python lists
    if isinstance(first, list):
        if dim == 0:
            res = []
            for t in tensors:
                res.extend(t)
            return res
        elif dim == 1:
            res = []
            for row_idx in range(len(first)):
                row = []
                for t in tensors:
                    row.extend(t[row_idx])
                res.append(row)
            return res
        raise NotImplementedError(f"List concatenation for dim {dim} not supported.")

    raise HeadReconstructionError(f"Unsupported tensor type for concatenation: {type(first)}")


class TensorSplitter:
    """
    Core splitting engine implementing Column-Parallel, Row-Parallel,
    and Attention-Head partitioning.
    """

    @staticmethod
    def split_along_dim(tensor: Any, dim: int, world_size: int, rank: int) -> Any:
        """
        Split a tensor along a specified dimension into world_size equal parts,
        returning the slice corresponding to rank.
        """
        shape = _get_shape(tensor)
        ndim = len(shape)
        norm_dim = dim if dim >= 0 else ndim + dim

        if norm_dim < 0 or norm_dim >= ndim:
            raise ParallelShapeMismatchError(
                f"Dimension {dim} out of range for tensor of shape {shape}"
            )

        dim_size = shape[norm_dim]
        if dim_size % world_size != 0:
            raise ParallelShapeMismatchError(
                f"Dimension size {dim_size} at dim {dim} is not divisible by world_size {world_size}"
            )

        chunk_size = dim_size // world_size
        start = rank * chunk_size
        end = start + chunk_size

        return _slice_dim(tensor, norm_dim, start, end)

    @classmethod
    def split_column(cls, weight: Any, world_size: int, rank: int, dim: int = 0) -> Any:
        """
        Column-parallel splitting of linear weight matrix.
        In PyTorch linear layers (out_features, in_features), out_features is dim 0.
        """
        return cls.split_along_dim(weight, dim=dim, world_size=world_size, rank=rank)

    @classmethod
    def split_row(cls, weight: Any, world_size: int, rank: int, dim: int = 1) -> Any:
        """
        Row-parallel splitting of linear weight matrix.
        In PyTorch linear layers (out_features, in_features), in_features is dim 1.
        """
        return cls.split_along_dim(weight, dim=dim, world_size=world_size, rank=rank)

    @classmethod
    def split_attention_heads(
        cls,
        weight: Any,
        num_heads: int,
        head_dim: int,
        world_size: int,
        rank: int,
        dim: int = 0,
    ) -> Any:
        """
        Split an attention weight tensor along head boundaries.
        Ensures that heads are not fragmented mid-dimension.
        """
        if num_heads % world_size != 0:
            raise AttentionHeadSplitError(
                f"num_heads ({num_heads}) must be divisible by world_size ({world_size})"
            )

        shape = _get_shape(weight)
        norm_dim = dim if dim >= 0 else len(shape) + dim
        expected_size = num_heads * head_dim

        if shape[norm_dim] != expected_size:
            raise AttentionHeadSplitError(
                f"Weight dim {dim} size {shape[norm_dim]} does not match num_heads * head_dim "
                f"({num_heads} * {head_dim} = {expected_size})"
            )

        heads_per_rank = num_heads // world_size
        start_head = rank * heads_per_rank
        end_head = start_head + heads_per_rank

        start_idx = start_head * head_dim
        end_idx = end_head * head_dim

        return _slice_dim(weight, norm_dim, start_idx, end_idx)


class TensorMerger:
    """Inverse operator that merges rank slices back into the unified monolithic tensor."""

    @staticmethod
    def merge_along_dim(slices: Sequence[Any], dim: int = 0) -> Any:
        """Concatenate a sequence of rank slices along dim to reconstruct full tensor."""
        return _concat_dim(slices, dim=dim)

    @classmethod
    def merge_column(cls, slices: Sequence[Any], dim: int = 0) -> Any:
        """Merge column-parallel slices back into full weight."""
        return cls.merge_along_dim(slices, dim=dim)

    @classmethod
    def merge_row(cls, slices: Sequence[Any], dim: int = 1) -> Any:
        """Merge row-parallel slices back into full weight."""
        return cls.merge_along_dim(slices, dim=dim)
