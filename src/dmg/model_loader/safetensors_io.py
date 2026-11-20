"""
Hugging Face Safetensors format integration, binary header parser,
multi-shard index resolver, and checkpoint serialization utilities.

Safetensors binary format specification:
- Bytes [0..8): 64-bit unsigned little-endian integer `N` (size of JSON header in bytes)
- Bytes [8..8+N): UTF-8 encoded JSON header mapping tensor names to:
    {"dtype": "F32", "shape": [d0, d1, ...], "data_offsets": [begin, end]}
  plus an optional "__metadata__" object.
- Bytes [8+N..EOF): Contiguous raw byte buffer of tensor payloads.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from dmg.core.exceptions import SafetensorsFormatError
from dmg.core.interfaces import TensorMetadata
from dmg.model_loader.config import dtype_byte_size, normalize_dtype

try:
    from safetensors.numpy import load_file as _st_load_file
    from safetensors.numpy import save_file as _st_save_file

    HAS_SAFETENSORS_LIB = True
except ImportError:  # pragma: no cover
    HAS_SAFETENSORS_LIB = False

MAX_HEADER_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB safety ceiling per Safetensors spec

SAFETENSORS_TO_NP_DTYPE: Dict[str, np.dtype[Any]] = {
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),  # Represented as raw 16-bit words when stored in NumPy
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I8": np.dtype("<i1"),
    "U8": np.dtype("<u1"),
}


@dataclass(frozen=True)
class SafetensorsHeader:
    """Parsed header of a single `.safetensors` file."""

    file_path: Path
    header_size: int
    data_base_offset: int
    tensors: Dict[str, TensorMetadata]
    custom_metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def total_payload_bytes(self) -> int:
        """Total byte size of all tensor payloads in this file."""
        if not self.tensors:
            return 0
        return max(meta.data_offsets[1] for meta in self.tensors.values())


def parse_safetensors_header(file_path: Union[str, Path]) -> SafetensorsHeader:
    """
    Parse only the binary header of a `.safetensors` file without loading
    the underlying tensor weights into memory.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise SafetensorsFormatError(f"Safetensors file does not exist: {path}")

    file_size = path.stat().st_size
    if file_size < 8:
        raise SafetensorsFormatError(
            f"File {path} is too small ({file_size} bytes) to contain a valid 8-byte Safetensors header length."
        )

    with path.open("rb") as fp:
        raw_len = fp.read(8)
        (header_size,) = struct.unpack("<Q", raw_len)

        if header_size == 0 or header_size > MAX_HEADER_SIZE_BYTES:
            raise SafetensorsFormatError(
                f"Invalid Safetensors header size {header_size} in {path}."
            )
        if 8 + header_size > file_size:
            raise SafetensorsFormatError(
                f"Safetensors header size ({header_size}) exceeds file size ({file_size}) in {path}."
            )

        raw_header = fp.read(header_size)

    try:
        header_dict = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsFormatError(
            f"Malformed UTF-8 JSON header in {path}: {exc}"
        ) from exc

    if not isinstance(header_dict, dict):
        raise SafetensorsFormatError(
            f"Safetensors header must be a JSON object in {path}."
        )

    data_base_offset = 8 + header_size
    payload_capacity = file_size - data_base_offset
    tensors: Dict[str, TensorMetadata] = {}
    custom_metadata: Dict[str, str] = {}

    for key, entry in header_dict.items():
        if key == "__metadata__":
            if isinstance(entry, dict):
                custom_metadata = {str(k): str(v) for k, v in entry.items()}
            continue

        if not isinstance(entry, dict):
            raise SafetensorsFormatError(
                f"Invalid metadata entry for tensor '{key}' in {path}."
            )

        try:
            dtype = normalize_dtype(str(entry["dtype"]))
            shape = tuple(int(dim) for dim in entry["shape"])
            offsets = entry["data_offsets"]
            begin, end = int(offsets[0]), int(offsets[1])
        except (KeyError, TypeError, ValueError) as exc:
            raise SafetensorsFormatError(
                f"Malformed descriptor for tensor '{key}' in {path}: {exc}"
            ) from exc

        if begin < 0 or end < begin or end > payload_capacity:
            raise SafetensorsFormatError(
                f"Out-of-bounds data_offsets [{begin}, {end}] for tensor '{key}' "
                f"(payload capacity={payload_capacity}) in {path}."
            )

        expected_bytes = 1
        for dim in shape:
            if dim < 0:
                raise SafetensorsFormatError(
                    f"Negative dimension in shape {shape} for tensor '{key}' in {path}."
                )
            expected_bytes *= dim
        expected_bytes *= dtype_byte_size(dtype)

        if (end - begin) != expected_bytes:
            raise SafetensorsFormatError(
                f"Byte size mismatch for tensor '{key}' in {path}: "
                f"offsets span {end - begin} bytes, but shape {shape} ({dtype}) requires {expected_bytes} bytes."
            )

        tensors[key] = TensorMetadata(
            name=key,
            shape=shape,
            dtype=dtype,
            data_offsets=(begin, end),
            file_path=str(path.resolve()),
        )

    return SafetensorsHeader(
        file_path=path.resolve(),
        header_size=header_size,
        data_base_offset=data_base_offset,
        tensors=tensors,
        custom_metadata=custom_metadata,
    )


class SafetensorsIndex:
    """
    Resolves single-file (`model.safetensors`) or multi-shard
    (`model.safetensors.index.json` / `model-00001-of-0000N.safetensors`)
    Hugging Face checkpoints into a unified tensor directory.
    """

    def __init__(self, checkpoint_path: Union[str, Path]) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.headers: Dict[Path, SafetensorsHeader] = {}
        self.tensor_map: Dict[str, TensorMetadata] = {}
        self.index_metadata: Dict[str, Any] = {}
        self._discover_and_parse()

    def _discover_and_parse(self) -> None:
        if not self.checkpoint_path.exists():
            raise SafetensorsFormatError(
                f"Checkpoint path does not exist: {self.checkpoint_path}"
            )

        shard_files: List[Path] = []
        if self.checkpoint_path.is_file():
            if self.checkpoint_path.name.endswith(".index.json"):
                shard_files = self._parse_index_json(self.checkpoint_path)
            elif self.checkpoint_path.suffix == ".safetensors":
                shard_files = [self.checkpoint_path]
            else:
                raise SafetensorsFormatError(
                    f"Unsupported checkpoint file extension: {self.checkpoint_path}"
                )
        else:
            index_file = self.checkpoint_path / "model.safetensors.index.json"
            single_file = self.checkpoint_path / "model.safetensors"
            if index_file.exists():
                shard_files = self._parse_index_json(index_file)
            elif single_file.exists():
                shard_files = [single_file]
            else:
                shard_files = sorted(self.checkpoint_path.glob("*.safetensors"))
                if not shard_files:
                    raise SafetensorsFormatError(
                        f"No .safetensors files found in directory: {self.checkpoint_path}"
                    )

        for shard_path in shard_files:
            resolved = shard_path.resolve()
            header = parse_safetensors_header(resolved)
            self.headers[resolved] = header
            for name, meta in header.tensors.items():
                if name in self.tensor_map:
                    raise SafetensorsFormatError(
                        f"Duplicate tensor '{name}' found across multiple shards: "
                        f"{self.tensor_map[name].file_path} and {resolved}"
                    )
                self.tensor_map[name] = meta

    def _parse_index_json(self, index_file: Path) -> List[Path]:
        try:
            raw = json.loads(index_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SafetensorsFormatError(
                f"Invalid JSON in shard index {index_file}: {exc}"
            ) from exc

        weight_map = raw.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise SafetensorsFormatError(
                f"Missing or empty 'weight_map' in shard index {index_file}"
            )
        self.index_metadata = raw.get("metadata", {})

        unique_files: List[Path] = []
        seen = set()
        for rel_file in weight_map.values():
            candidate = (index_file.parent / rel_file).resolve()
            if not candidate.exists():
                raise SafetensorsFormatError(
                    f"Shard file '{rel_file}' referenced by {index_file} does not exist."
                )
            if candidate not in seen:
                seen.add(candidate)
                unique_files.append(candidate)
        return unique_files

    def get_metadata(self, tensor_name: str) -> TensorMetadata:
        """Retrieve `TensorMetadata` for a named parameter."""
        if tensor_name not in self.tensor_map:
            raise KeyError(
                f"Tensor '{tensor_name}' not found in checkpoint {self.checkpoint_path}"
            )
        return self.tensor_map[tensor_name]

    def list_tensors(self) -> List[str]:
        """Return all tensor names in deterministic order."""
        return list(self.tensor_map.keys())

    @property
    def num_shards(self) -> int:
        """Number of physical `.safetensors` files backing this checkpoint."""
        return len(self.headers)


def save_safetensors_file(
    tensors: Dict[str, np.ndarray[Any, Any]],
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Serialize a dictionary of NumPy arrays into a `.safetensors` file.
    Uses `safetensors.numpy.save_file` when installed, with a spec-compliant
    deterministic binary writer fallback.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if HAS_SAFETENSORS_LIB and all(
        arr.dtype in (np.float32, np.float16, np.int64, np.int32, np.int8, np.uint8)
        for arr in tensors.values()
    ):
        contiguous = {k: np.ascontiguousarray(v) for k, v in tensors.items()}
        _st_save_file(contiguous, str(out), metadata=metadata)
        return out

    # Spec-compliant binary writer (also supports custom raw BF16 uint16 buffers)
    header_obj: Dict[str, Any] = {}
    if metadata:
        header_obj["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    offset = 0
    encoded_buffers: List[bytes] = []
    for name, arr in tensors.items():
        contig = np.ascontiguousarray(arr)
        dtype_str = normalize_dtype(str(contig.dtype))
        raw_bytes = contig.tobytes()
        begin = offset
        end = offset + len(raw_bytes)
        header_obj[name] = {
            "dtype": dtype_str,
            "shape": list(contig.shape),
            "data_offsets": [begin, end],
        }
        encoded_buffers.append(raw_bytes)
        offset = end

    header_bytes = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
    # Pad header to 8-byte alignment with spaces per HuggingFace convention
    padding = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * padding

    with out.open("wb") as fp:
        fp.write(struct.pack("<Q", len(header_bytes)))
        fp.write(header_bytes)
        for buf in encoded_buffers:
            fp.write(buf)

    return out
