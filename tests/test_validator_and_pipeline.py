"""
Unit and integration tests for Week 2:
- Tensor shape, numerical finiteness (NaN/Inf), and SHA-256/CRC32 integrity validation (`validator.py`)
- Optimized `LLMWeightLoader` with LRU caching, thread-pool prefetching, dtype casting, and end-to-end cluster preparation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from dmg.core.exceptions import IntegrityValidationError, ShapeMismatchError
from dmg.core.interfaces import WorkerShardTarget
from dmg.model_loader.config import LLMModelConfig
from dmg.model_loader.loader import LLMWeightLoader
from dmg.model_loader.mmap_loader import MemoryMappedWeightStore
from dmg.model_loader.sharding import WeightDistributor
from dmg.model_loader.validator import TensorIntegrityValidator


class TestTensorIntegrityValidator:
    """Tests for `TensorIntegrityValidator`."""

    def test_full_checkpoint_validation_passes(
        self,
        multi_shard_checkpoint: Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]],
    ) -> None:
        ckpt_dir, config, _ = multi_shard_checkpoint
        validator = TensorIntegrityValidator(config)
        with MemoryMappedWeightStore(ckpt_dir) as store:
            report = validator.validate_store(store, disallow_all_zeros=True)
            assert report.passed
            assert not report.missing_tensors
            assert not report.unexpected_tensors
            assert not report.failed_tensors
            assert report.total_tensors_checked == len(config.expected_tensor_shapes())

    def test_detects_nan_and_inf_weights(
        self, tiny_model_config: LLMModelConfig
    ) -> None:
        validator = TensorIntegrityValidator(tiny_model_config)
        corrupted = np.ones((tiny_model_config.vocab_size, tiny_model_config.hidden_size), dtype=np.float32)
        corrupted[0, 0] = np.nan

        with pytest.raises(IntegrityValidationError, match="Numerical corruption"):
            validator.validate_single_tensor("model.embed_tokens.weight", corrupted)

    def test_detects_shape_mismatch(
        self, tiny_model_config: LLMModelConfig
    ) -> None:
        validator = TensorIntegrityValidator(tiny_model_config)
        wrong_shape = np.ones((10, 10), dtype=np.float32)

        with pytest.raises(ShapeMismatchError, match="Shape mismatch"):
            validator.validate_single_tensor("model.embed_tokens.weight", wrong_shape)

    def test_validates_distributed_packages_and_detects_tampering(
        self,
        single_shard_checkpoint: Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]],
    ) -> None:
        ckpt_dir, config, _ = single_shard_checkpoint
        validator = TensorIntegrityValidator(config)
        workers = [
            WorkerShardTarget(node_id=f"node-{r}", rank=r, world_size=4)
            for r in range(4)
        ]

        with MemoryMappedWeightStore(ckpt_dir) as store:
            distributor = WeightDistributor(config, store)
            packages = distributor.prepare_cluster_distribution(workers)
            assert validator.validate_distributed_packages(packages, store=store)

            # Tamper with a weight in rank 2's shard package
            packages[2].tensors["model.layers.0.self_attn.q_proj.weight"][0, 0] += 999.0
            with pytest.raises(IntegrityValidationError, match="SHA-256 checksum mismatch"):
                validator.validate_distributed_packages(packages, store=store)


class TestOptimizedEndToEndLoaderPipeline:
    """Integration tests for Week 2 optimized `LLMWeightLoader`."""

    def test_prefetching_caching_cast_and_distribution(
        self,
        multi_shard_checkpoint: Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]],
    ) -> None:
        ckpt_dir, config, _ = multi_shard_checkpoint
        workers = [
            WorkerShardTarget(node_id=f"node-{r}", rank=r, world_size=2)
            for r in range(2)
        ]

        with LLMWeightLoader(
            ckpt_dir,
            cache_capacity_tensors=16,
            prefetch_workers=2,
            target_dtype="F16",
        ) as loader:
            # Validate checkpoint before distributing
            report = loader.validate_checkpoint()
            assert report.passed

            # Prefetch layer 0 and 1 in background thread pool
            loader.prefetch_layers([0, 1])
            layer0 = loader.load_layer(0)
            assert layer0.q_proj.dtype == np.float16
            assert loader.cache_stats()["cached_tensors"] > 0

            # Prepare and verify cluster shards end-to-end
            packages = loader.prepare_distributed_shards(workers, validate=True)
            assert len(packages) == 2
            assert packages[0].target.node_id == "node-0"
            assert packages[1].target.node_id == "node-1"
