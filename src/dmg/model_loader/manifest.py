"""
Weight Checkpoint Integrity Manifest Generator, Signer, and Verification CLI.

Generates cryptographic manifest specifications containing SHA-256 and BLAKE2b hashes,
tensor-parallel partition shapes, and parameter fingerprints across all model shards
for automated pre-flight checks in Member 1's distributed cluster.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from dmg.core.exceptions import IntegrityValidationError
from dmg.model_loader.config import LLMModelConfig
from dmg.model_loader.mmap_loader import MemoryMappedWeightStore
from dmg.model_loader.sharding import PartitionStrategy, WeightDistributor


@dataclass(frozen=True)
class TensorFingerprint:
    """Cryptographic and architectural fingerprint for a single model parameter tensor."""

    name: str
    shape: List[int]
    dtype: str
    num_bytes: int
    sha256: str
    blake2b: str
    strategy: str
    split_dim: Optional[int]


@dataclass
class CheckpointManifest:
    """Signed, verifiable manifest of an entire Safetensors checkpoint."""

    model_name: str
    manifest_version: str
    total_tensors: int
    total_parameters: int
    total_memory_bytes: int
    tensors: Dict[str, TensorFingerprint] = field(default_factory=dict)
    root_hash: str = ""

    def compute_root_hash(self) -> str:
        """Compute Merkle-style root hash combining all tensor fingerprints in deterministic key order."""
        hasher = hashlib.sha256()
        for name in sorted(self.tensors.keys()):
            fp = self.tensors[name]
            hasher.update(f"{fp.name}:{fp.sha256}:{fp.blake2b}".encode("utf-8"))
        self.root_hash = hasher.hexdigest()
        return self.root_hash

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to serializable dictionary."""
        return asdict(self)

    def save_json(self, output_path: Union[str, Path]) -> Path:
        """Save manifest to a JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not self.root_hash:
            self.compute_root_hash()
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "CheckpointManifest":
        """Load and parse manifest from JSON."""
        target = Path(json_path)
        if not target.exists():
            raise IntegrityValidationError(f"Manifest file not found: {target}")
        data = json.loads(target.read_text(encoding="utf-8"))
        tensors = {
            k: TensorFingerprint(**v) for k, v in data.get("tensors", {}).items()
        }
        return cls(
            model_name=data["model_name"],
            manifest_version=data.get("manifest_version", "1.0"),
            total_tensors=data["total_tensors"],
            total_parameters=data["total_parameters"],
            total_memory_bytes=data["total_memory_bytes"],
            tensors=tensors,
            root_hash=data.get("root_hash", ""),
        )


class ManifestBuilder:
    """Builds and verifies cryptographically signed `CheckpointManifest` objects."""

    def __init__(self, config: LLMModelConfig, store: MemoryMappedWeightStore) -> None:
        self.config = config
        self.store = store

    def build_manifest(self) -> CheckpointManifest:
        """Scan all tensors in store and generate cryptographic fingerprints."""
        fingerprints: Dict[str, TensorFingerprint] = {}
        total_params = 0
        total_bytes = 0

        for name in sorted(self.store.list_tensors()):
            meta = self.store.get_tensor_metadata(name)
            raw_view = self.store.get_raw_bytes(name)

            sha256_hex = hashlib.sha256(raw_view).hexdigest()
            blake2b_hex = hashlib.blake2b(raw_view).hexdigest()

            strategy, split_dim = WeightDistributor.classify_tensor_strategy(name)

            prod = 1
            for d in meta.shape:
                prod *= d
            total_params += prod
            total_bytes += meta.num_bytes

            fingerprints[name] = TensorFingerprint(
                name=name,
                shape=list(meta.shape),
                dtype=meta.dtype,
                num_bytes=meta.num_bytes,
                sha256=sha256_hex,
                blake2b=blake2b_hex,
                strategy=strategy.value if hasattr(strategy, "value") else str(strategy),
                split_dim=split_dim,
            )

        manifest = CheckpointManifest(
            model_name=self.config.model_name_or_path,
            manifest_version="1.0",
            total_tensors=len(fingerprints),
            total_parameters=total_params,
            total_memory_bytes=total_bytes,
            tensors=fingerprints,
        )
        manifest.compute_root_hash()
        return manifest

    def verify_against_manifest(self, manifest: CheckpointManifest) -> bool:
        """
        Verify that all tensors in the local memory-mapped store match
        the cryptographic signatures in `manifest`.
        """
        current = self.build_manifest()
        if current.root_hash != manifest.root_hash:
            raise IntegrityValidationError(
                f"Root hash mismatch: expected {manifest.root_hash}, calculated {current.root_hash}"
            )
        return True
