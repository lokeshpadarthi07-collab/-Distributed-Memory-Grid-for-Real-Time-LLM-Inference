"""
Tensor shape, dtype, numerical sanity (NaN/Inf), and cryptographic checksum
integrity validation module (Week 2).
"""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dmg.core.exceptions import IntegrityValidationError, ShapeMismatchError
from dmg.model_loader.config import LLMModelConfig, normalize_dtype
from dmg.model_loader.mmap_loader import MemoryMappedWeightStore
from dmg.model_loader.sharding import NodeShardPackage, PartitionStrategy


@dataclass(frozen=True)
class TensorValidationRecord:
    """Detailed validation result for an individual tensor."""

    tensor_name: str
    expected_shape: Optional[Tuple[int, ...]]
    actual_shape: Tuple[int, ...]
    dtype: str
    has_nan: bool
    has_inf: bool
    is_all_zero: bool
    sha256: str
    crc32: int
    passed: bool
    error_message: str = ""


@dataclass
class CheckpointValidationReport:
    """Aggregate report produced by `TensorIntegrityValidator`."""

    model_name: str
    total_tensors_checked: int
    passed: bool
    missing_tensors: List[str] = field(default_factory=list)
    unexpected_tensors: List[str] = field(default_factory=list)
    records: Dict[str, TensorValidationRecord] = field(default_factory=dict)

    @property
    def failed_tensors(self) -> List[str]:
        return [name for name, rec in self.records.items() if not rec.passed]


class TensorIntegrityValidator:
    """
    Validates LLM checkpoint weights and distributed node shards for:
    1. Exact tensor shape & dtype conformance against `LLMModelConfig`.
    2. Complete parameter coverage (detecting missing or unexpected tensors).
    3. Numerical stability (absence of NaN / Inf values, optional zero-tensor detection).
    4. Cryptographic (SHA-256) and fast (CRC32) byte-level integrity across worker nodes.
    """

    def __init__(self, config: LLMModelConfig) -> None:
        self.config = config
        self.expected_shapes = config.expected_tensor_shapes()

    @staticmethod
    def compute_sha256(data: bytes | memoryview | np.ndarray[Any, Any]) -> str:
        """Compute deterministic SHA-256 hex digest over raw tensor bytes."""
        if isinstance(data, np.ndarray):
            buf = np.ascontiguousarray(data).tobytes()
            return hashlib.sha256(buf).hexdigest()
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def compute_crc32(data: bytes | memoryview | np.ndarray[Any, Any]) -> int:
        """Compute unsigned 32-bit CRC32 checksum over raw tensor bytes."""
        if isinstance(data, np.ndarray):
            buf = np.ascontiguousarray(data).tobytes()
            return zlib.crc32(buf) & 0xFFFFFFFF
        return zlib.crc32(data) & 0xFFFFFFFF

    def validate_single_tensor(
        self,
        tensor_name: str,
        tensor: np.ndarray[Any, Any],
        *,
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_dtype: Optional[str] = None,
        check_nan_inf: bool = True,
        disallow_all_zeros: bool = False,
        raise_on_error: bool = True,
    ) -> TensorValidationRecord:
        """
        Validate a single loaded tensor's shape, dtype, numerical finiteness,
        and compute its SHA-256 and CRC32 digests.
        """
        target_shape = (
            expected_shape
            if expected_shape is not None
            else self.expected_shapes.get(tensor_name)
        )
        actual_shape = tuple(tensor.shape)
        actual_dtype = normalize_dtype(str(tensor.dtype))

        if target_shape is not None and actual_shape != target_shape:
            msg = (
                f"Shape mismatch for tensor '{tensor_name}': "
                f"expected {target_shape}, got {actual_shape}"
            )
            if raise_on_error:
                raise ShapeMismatchError(msg)
            return TensorValidationRecord(
                tensor_name=tensor_name,
                expected_shape=target_shape,
                actual_shape=actual_shape,
                dtype=actual_dtype,
                has_nan=False,
                has_inf=False,
                is_all_zero=False,
                sha256="",
                crc32=0,
                passed=False,
                error_message=msg,
            )

        if expected_dtype is not None:
            norm_expected_dtype = normalize_dtype(expected_dtype)
            if actual_dtype != norm_expected_dtype:
                msg = (
                    f"Dtype mismatch for tensor '{tensor_name}': "
                    f"expected {norm_expected_dtype}, got {actual_dtype}"
                )
                if raise_on_error:
                    raise ShapeMismatchError(msg)
                return TensorValidationRecord(
                    tensor_name=tensor_name,
                    expected_shape=target_shape,
                    actual_shape=actual_shape,
                    dtype=actual_dtype,
                    has_nan=False,
                    has_inf=False,
                    is_all_zero=False,
                    sha256="",
                    crc32=0,
                    passed=False,
                    error_message=msg,
                )

        has_nan = False
        has_inf = False
        if check_nan_inf and np.issubdtype(tensor.dtype, np.floating):
            has_nan = bool(np.isnan(tensor).any())
            has_inf = bool(np.isinf(tensor).any())
            if has_nan or has_inf:
                msg = (
                    f"Numerical corruption detected in tensor '{tensor_name}': "
                    f"has_nan={has_nan}, has_inf={has_inf}"
                )
                if raise_on_error:
                    raise IntegrityValidationError(msg)
                return TensorValidationRecord(
                    tensor_name=tensor_name,
                    expected_shape=target_shape,
                    actual_shape=actual_shape,
                    dtype=actual_dtype,
                    has_nan=has_nan,
                    has_inf=has_inf,
                    is_all_zero=False,
                    sha256="",
                    crc32=0,
                    passed=False,
                    error_message=msg,
                )

        is_all_zero = bool(tensor.size > 0 and not np.any(tensor))
        if disallow_all_zeros and is_all_zero:
            msg = f"Tensor '{tensor_name}' contains all zeros (uninitialized weights)."
            if raise_on_error:
                raise IntegrityValidationError(msg)
            return TensorValidationRecord(
                tensor_name=tensor_name,
                expected_shape=target_shape,
                actual_shape=actual_shape,
                dtype=actual_dtype,
                has_nan=has_nan,
                has_inf=has_inf,
                is_all_zero=is_all_zero,
                sha256="",
                crc32=0,
                passed=False,
                error_message=msg,
            )

        raw_bytes = np.ascontiguousarray(tensor).tobytes()
        sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
        crc32_val = zlib.crc32(raw_bytes) & 0xFFFFFFFF

        return TensorValidationRecord(
            tensor_name=tensor_name,
            expected_shape=target_shape,
            actual_shape=actual_shape,
            dtype=actual_dtype,
            has_nan=has_nan,
            has_inf=has_inf,
            is_all_zero=is_all_zero,
            sha256=sha256_hex,
            crc32=crc32_val,
            passed=True,
        )

    def validate_store(
        self,
        store: MemoryMappedWeightStore,
        *,
        strict_coverage: bool = True,
        check_nan_inf: bool = True,
        disallow_all_zeros: bool = False,
        raise_on_error: bool = True,
    ) -> CheckpointValidationReport:
        """
        Validate all tensors in a `MemoryMappedWeightStore` against `self.config`.
        """
        available = set(store.list_tensors())
        expected = set(self.expected_shapes.keys())

        missing = sorted(expected - available)
        unexpected = sorted(available - expected)

        if strict_coverage and missing and raise_on_error:
            raise ShapeMismatchError(
                f"Checkpoint is missing {len(missing)} required tensor(s): {missing[:5]}"
            )
        if strict_coverage and unexpected and raise_on_error:
            raise ShapeMismatchError(
                f"Checkpoint contains {len(unexpected)} unexpected tensor(s): {unexpected[:5]}"
            )

        records: Dict[str, TensorValidationRecord] = {}
        for name in store.list_tensors():
            arr = store.get_tensor(name, copy=False)
            records[name] = self.validate_single_tensor(
                name,
                arr,
                expected_shape=self.expected_shapes.get(name),
                check_nan_inf=check_nan_inf,
                disallow_all_zeros=disallow_all_zeros,
                raise_on_error=raise_on_error,
            )

        all_passed = (
            all(r.passed for r in records.values())
            and (not strict_coverage or (not missing and not unexpected))
        )
        return CheckpointValidationReport(
            model_name=self.config.model_name_or_path,
            total_tensors_checked=len(records),
            passed=all_passed,
            missing_tensors=missing,
            unexpected_tensors=unexpected,
            records=records,
        )

    def validate_distributed_packages(
        self,
        packages: Dict[int, NodeShardPackage],
        store: Optional[MemoryMappedWeightStore] = None,
    ) -> bool:
        """
        Verify that all distributed `NodeShardPackage` instances across ranks:
        1. Have matching SHA-256 checksums for every shard slice.
        2. Have valid partitioned dimensions summing to the full model tensor shape.
        3. Reconstruct bit-for-bit to the source checkpoint tensors when `store` is provided.
        """
        if not packages:
            raise IntegrityValidationError("No node packages provided for validation.")

        ranks = sorted(packages.keys())
        world_size = len(ranks)

        for rank, pkg in packages.items():
            for name, desc in pkg.descriptors.items():
                if name not in pkg.tensors:
                    raise IntegrityValidationError(
                        f"Missing tensor payload '{name}' in rank {rank} package."
                    )
                arr = pkg.tensors[name]
                if tuple(arr.shape) != desc.shard_shape:
                    raise ShapeMismatchError(
                        f"Shard shape mismatch for '{name}' on rank {rank}: "
                        f"descriptor={desc.shard_shape}, array={arr.shape}"
                    )
                if desc.sha256:
                    actual_sha = self.compute_sha256(arr)
                    if actual_sha != desc.sha256:
                        raise IntegrityValidationError(
                            f"SHA-256 checksum mismatch for '{name}' on rank {rank}: "
                            f"expected {desc.sha256}, got {actual_sha}"
                        )

        first_pkg = packages[ranks[0]]
        for name, first_desc in first_pkg.descriptors.items():
            shards = [
                (packages[r].descriptors[name], packages[r].tensors[name])
                for r in ranks
            ]
            if first_desc.strategy != PartitionStrategy.REPLICATED and first_desc.split_dim is not None:
                split_dim = first_desc.split_dim
                total_split = sum(arr.shape[split_dim] for _, arr in shards)
                if total_split != first_desc.source_shape[split_dim]:
                    raise ShapeMismatchError(
                        f"Partitioned dimension {split_dim} for '{name}' sums to "
                        f"{total_split} across {world_size} ranks, expected {first_desc.source_shape[split_dim]}"
                    )

            if store is not None:
                original = store.get_tensor(name, copy=False)
                if first_desc.strategy == PartitionStrategy.REPLICATED or first_desc.split_dim is None:
                    for r, (_, shard_arr) in zip(ranks, shards):
                        if not np.array_equal(original, shard_arr):
                            raise IntegrityValidationError(
                                f"Replicated tensor '{name}' on rank {r} differs from source checkpoint."
                            )
                else:
                    reconstructed = np.concatenate(
                        [arr for _, arr in shards], axis=first_desc.split_dim
                    )
                    if not np.array_equal(original, reconstructed):
                        raise IntegrityValidationError(
                            f"Reconstructed tensor '{name}' across {world_size} ranks "
                            f"does not match source checkpoint."
                        )

        return True
