"""
Memory-mapped (mmap) zero-copy model weight store for `.safetensors` files.

Maps `.safetensors` files directly into the process virtual address space
using read-only OS page mappings (`mmap.ACCESS_READ`), avoiding full-file
heap copies and allowing instant zero-copy NumPy (and PyTorch-compatible)
tensor views over arbitrary byte ranges or sub-tensor slices.
"""

from __future__ import annotations

import mmap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from dmg.core.exceptions import MemoryMapError
from dmg.core.interfaces import TensorMetadata
from dmg.model_loader.safetensors_io import (
    SAFETENSORS_TO_NP_DTYPE,
    SafetensorsHeader,
    SafetensorsIndex,
    parse_safetensors_header,
)


@dataclass(frozen=True)
class MemoryMapStats:
    """Telemetry snapshot of active virtual memory mappings."""

    open_files: int
    total_mapped_bytes: int
    total_tensors_indexed: int
    tensor_loads_served: int


class _MappedSafetensorsFile:
    """Internal handle managing the file descriptor and `mmap.mmap` object for one shard."""

    def __init__(self, header: SafetensorsHeader) -> None:
        self.header = header
        self.file_path = header.file_path
        self._fp: Optional[Any] = None
        self._mmap: Optional[mmap.mmap] = None
        self._open()

    def _open(self) -> None:
        try:
            self._fp = self.file_path.open("rb")
            self._mmap = mmap.mmap(
                self._fp.fileno(),
                length=0,
                access=mmap.ACCESS_READ,
            )
            # Provide OS sequential/random read hint on POSIX when available
            if hasattr(self._mmap, "madvise") and hasattr(mmap, "MADV_WILLNEED"):
                try:
                    self._mmap.madvise(mmap.MADV_WILLNEED)
                except OSError:
                    pass
        except OSError as exc:
            self.close()
            raise MemoryMapError(
                f"Failed to memory-map Safetensors shard {self.file_path}: {exc}"
            ) from exc

    @property
    def is_open(self) -> bool:
        return self._mmap is not None and not self._mmap.closed

    @property
    def mapped_bytes(self) -> int:
        if self._mmap is None or self._mmap.closed:
            return 0
        return len(self._mmap)

    def get_tensor_view(
        self,
        meta: TensorMetadata,
        copy: bool = False,
    ) -> np.ndarray[Any, Any]:
        """
        Return a read-only zero-copy NumPy array backed directly by the OS
        memory map for `meta` (unless `copy=True` is requested).
        """
        if self._mmap is None or self._mmap.closed:
            raise MemoryMapError(
                f"Cannot read tensor '{meta.name}' from closed memory map {self.file_path}"
            )

        np_dtype = SAFETENSORS_TO_NP_DTYPE.get(meta.dtype)
        if np_dtype is None:
            raise MemoryMapError(
                f"Unsupported dtype '{meta.dtype}' for memory-mapped tensor '{meta.name}'"
            )

        abs_start = self.header.data_base_offset + meta.data_offsets[0]
        abs_end = self.header.data_base_offset + meta.data_offsets[1]

        if abs_end > len(self._mmap):
            raise MemoryMapError(
                f"Tensor '{meta.name}' byte range [{abs_start}, {abs_end}) exceeds "
                f"mapped file length {len(self._mmap)} in {self.file_path}"
            )

        view = memoryview(self._mmap)[abs_start:abs_end]
        arr = np.frombuffer(view, dtype=np_dtype).reshape(meta.shape)
        if copy:
            return arr.copy()
        return arr

    def get_raw_bytes(self, meta: TensorMetadata) -> memoryview:
        """Return a zero-copy `memoryview` of the raw tensor payload bytes."""
        if self._mmap is None or self._mmap.closed:
            raise MemoryMapError(
                f"Cannot slice raw bytes for '{meta.name}' from closed memory map {self.file_path}"
            )
        abs_start = self.header.data_base_offset + meta.data_offsets[0]
        abs_end = self.header.data_base_offset + meta.data_offsets[1]
        return memoryview(self._mmap)[abs_start:abs_end]

    def close(self) -> None:
        """Release the `mmap` buffer and close the underlying file descriptor."""
        if self._mmap is not None:
            try:
                if not self._mmap.closed:
                    self._mmap.close()
            except (BufferError, ValueError):
                # Exported zero-copy buffers may still be alive; let GC finalize mmap
                pass
            finally:
                self._mmap = None

        if self._fp is not None:
            try:
                self._fp.close()
            finally:
                self._fp = None


class MemoryMappedWeightStore:
    """
    Enterprise memory-mapped weight store managing one or more `.safetensors`
    shards with zero-copy tensor views, 2D row/column slicing for distributed
    tensor parallelism, and deterministic cleanup via context manager.
    """

    def __init__(self, checkpoint_path: Union[str, Path]) -> None:
        self.index = SafetensorsIndex(checkpoint_path)
        self._handles: Dict[Path, _MappedSafetensorsFile] = {}
        self._loads_served: int = 0
        self._open_all_shards()

    def _open_all_shards(self) -> None:
        try:
            for shard_path, header in self.index.headers.items():
                self._handles[shard_path] = _MappedSafetensorsFile(header)
        except Exception:
            self.close()
            raise

    @property
    def is_open(self) -> bool:
        return bool(self._handles) and all(
            h.is_open for h in self._handles.values()
        )

    def get_tensor_metadata(self, tensor_name: str) -> TensorMetadata:
        """Return `TensorMetadata` for `tensor_name`."""
        return self.index.get_metadata(tensor_name)

    def list_tensors(self) -> List[str]:
        """List all tensor names available across all mapped shards."""
        return self.index.list_tensors()

    def get_tensor(
        self,
        tensor_name: str,
        *,
        copy: bool = False,
    ) -> np.ndarray[Any, Any]:
        """
        Retrieve a tensor by name as a zero-copy memory-mapped NumPy ndarray
        (or a writable heap copy if `copy=True`).
        """
        meta = self.index.get_metadata(tensor_name)
        handle = self._get_handle(meta)
        self._loads_served += 1
        return handle.get_tensor_view(meta, copy=copy)

    def get_tensor_slice(
        self,
        tensor_name: str,
        *,
        dim: int = 0,
        start: int = 0,
        end: Optional[int] = None,
        copy: bool = False,
    ) -> np.ndarray[Any, Any]:
        """
        Load a slice `[start:end]` along axis `dim` directly from the memory-mapped
        tensor. Used by Member 3 (Attention head splitting) and Member 4 (FFN splitting).
        """
        arr = self.get_tensor(tensor_name, copy=False)
        if dim < 0 or dim >= arr.ndim:
            raise MemoryMapError(
                f"Invalid slice dimension {dim} for tensor '{tensor_name}' with shape {arr.shape}"
            )
        dim_size = arr.shape[dim]
        actual_end = dim_size if end is None else end
        if start < 0 or actual_end < start or actual_end > dim_size:
            raise MemoryMapError(
                f"Invalid slice bounds [{start}:{actual_end}] along dim {dim} "
                f"(size {dim_size}) for tensor '{tensor_name}'"
            )

        slicer: List[slice] = [slice(None)] * arr.ndim
        slicer[dim] = slice(start, actual_end)
        sliced = arr[tuple(slicer)]
        if copy:
            return np.ascontiguousarray(sliced)
        return sliced

    def get_raw_bytes(self, tensor_name: str) -> memoryview:
        """Return a zero-copy `memoryview` of the raw bytes for `tensor_name`."""
        meta = self.index.get_metadata(tensor_name)
        handle = self._get_handle(meta)
        self._loads_served += 1
        return handle.get_raw_bytes(meta)

    def stats(self) -> MemoryMapStats:
        """Return current memory-mapping statistics."""
        open_count = sum(1 for h in self._handles.values() if h.is_open)
        mapped_bytes = sum(h.mapped_bytes for h in self._handles.values())
        return MemoryMapStats(
            open_files=open_count,
            total_mapped_bytes=mapped_bytes,
            total_tensors_indexed=len(self.index.tensor_map),
            tensor_loads_served=self._loads_served,
        )

    def _get_handle(self, meta: TensorMetadata) -> _MappedSafetensorsFile:
        shard_path = Path(meta.file_path).resolve()
        handle = self._handles.get(shard_path)
        if handle is None or not handle.is_open:
            raise MemoryMapError(
                f"Memory-mapped shard for tensor '{meta.name}' is not open: {shard_path}"
            )
        return handle

    def close(self) -> None:
        """Close all memory-mapped shard handles."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> "MemoryMappedWeightStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
