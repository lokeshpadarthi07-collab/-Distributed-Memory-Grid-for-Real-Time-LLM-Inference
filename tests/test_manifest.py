"""
Unit tests for CheckpointManifest and cryptographic integrity verification (`manifest.py`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from dmg.core.exceptions import IntegrityValidationError
from dmg.model_loader.config import LLMModelConfig
from dmg.model_loader.manifest import CheckpointManifest, ManifestBuilder
from dmg.model_loader.mmap_loader import MemoryMappedWeightStore


def test_manifest_builder_and_verification(
    tmp_path: Path,
    single_shard_checkpoint: Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]],
) -> None:
    ckpt_dir, config, weights = single_shard_checkpoint

    with MemoryMappedWeightStore(ckpt_dir) as store:
        builder = ManifestBuilder(config, store)
        manifest = builder.build_manifest()

        assert manifest.total_tensors == len(weights)
        assert manifest.total_memory_bytes > 0
        assert len(manifest.root_hash) == 64
        assert builder.verify_against_manifest(manifest)

        # Test JSON serialization roundtrip
        manifest_file = tmp_path / "checkpoint_manifest.json"
        manifest.save_json(manifest_file)
        loaded = CheckpointManifest.from_json(manifest_file)

        assert loaded.root_hash == manifest.root_hash
        assert loaded.total_tensors == manifest.total_tensors

        # Tampering check
        tampered_manifest = CheckpointManifest(
            model_name=manifest.model_name,
            manifest_version=manifest.manifest_version,
            total_tensors=manifest.total_tensors,
            total_parameters=manifest.total_parameters,
            total_memory_bytes=manifest.total_memory_bytes,
            tensors=manifest.tensors,
            root_hash="0000000000000000000000000000000000000000000000000000000000000000",
        )
        with pytest.raises(IntegrityValidationError, match="Root hash mismatch"):
            builder.verify_against_manifest(tampered_manifest)
