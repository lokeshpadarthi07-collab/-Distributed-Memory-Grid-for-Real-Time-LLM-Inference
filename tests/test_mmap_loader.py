"""
Unit tests for Week 1:
- Memory-mapped zero-copy weight loading (`mmap_loader.py`)
- Foundational LLM weight loader (`loader.py`)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from dmg.core.exceptions import MemoryMapError
from dmg.model_loader.config import LLMModelConfig
from dmg.model_loader.loader import LLMWeightLoader
from dmg.model_loader.mmap_loader import MemoryMappedWeightStore


class TestMemoryMappedWeightStore:
    """Tests for `MemoryMappedWeightStore`."""

    def test_zero_copy_tensor_loading(
        self,
        single_shard_checkpoint: Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]],
    ) -> None:
        ckpt_dir, _, expected_weights = single_shard_checkpoint
        with MemoryMappedWeightStore(ckpt_dir) as store:
            assert store.is_open
            for name, expected in expected_weights.items():
                loaded = store.get_tensor(name, copy=False)
                assert loaded.shape == expected.shape
                assert np.allclose(loaded, expected)
                # Zero-copy view over read-only mmap is non-writable
                assert not loaded.flags.writeable

            stats = store.stats()
            assert stats.open_files == 1
            assert stats.tensor_loads_served == len(expected_weights)
            assert stats.total_mapped_bytes > 0

        assert not store.is_open

    def test_tensor_axis_slicing(
        self,
        single_shard_checkpoint: Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]],
    ) -> None:
        ckpt_dir, _, expected_weights = single_shard_checkpoint
        target = "model.layers.0.self_attn.q_proj.weight"
        full = expected_weights[target]

        with MemoryMappedWeightStore(ckpt_dir) as store:
            # Slice along axis 0 (column-parallel output dimension)
            row_slice = store.get_tensor_slice(target, dim=0, start=16, end=32)
            assert np.array_equal(row_slice, full[16:32, :])

            # Slice along axis 1 (row-parallel input dimension)
            col_slice = store.get_tensor_slice(target, dim=1, start=0, end=16, copy=True)
            assert np.array_equal(col_slice, full[:, 0:16])
            assert col_slice.flags.c_contiguous

            with pytest.raises(MemoryMapError):
                store.get_tensor_slice(target, dim=0, start=32, end=16)

    def test_foundational_llm_weight_loader(
        self,
        multi_shard_checkpoint: Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]],
    ) -> None:
        ckpt_dir, config, expected_weights = multi_shard_checkpoint
        with LLMWeightLoader(ckpt_dir) as loader:
            embeddings = loader.load_embeddings()
            assert np.array_equal(
                embeddings["embed_tokens"], expected_weights["model.embed_tokens.weight"]
            )
            assert np.array_equal(
                embeddings["lm_head"], expected_weights["lm_head.weight"]
            )

            layers = list(loader.iter_layers())
            assert len(layers) == config.num_hidden_layers
            assert np.array_equal(
                layers[0].q_proj,
                expected_weights["model.layers.0.self_attn.q_proj.weight"],
            )
            assert np.array_equal(
                layers[1].down_proj,
                expected_weights["model.layers.1.mlp.down_proj.weight"],
            )
