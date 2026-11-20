"""
Unit tests for Week 2:
- Model weight distribution and tensor-parallel sharding across worker nodes (`sharding.py`)
- Attention head splitting (Member 3 integration) and FFN layer splitting (Member 4 integration)
- Lossless tensor reconstruction across worker shards
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from dmg.core.exceptions import ShardDistributionError
from dmg.core.interfaces import WorkerShardTarget
from dmg.model_loader.config import LLMModelConfig
from dmg.model_loader.mmap_loader import MemoryMappedWeightStore
from dmg.model_loader.sharding import PartitionStrategy, WeightDistributor


class TestWeightDistributionAndSharding:
    """Tests for `WeightDistributor` and `NodeShardPackage`."""

    def test_attention_head_sharding_for_member_3(
        self,
        single_shard_checkpoint: Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]],
    ) -> None:
        ckpt_dir, config, expected_weights = single_shard_checkpoint
        world_size = 4
        with MemoryMappedWeightStore(ckpt_dir) as store:
            distributor = WeightDistributor(config, store)

            q_shards = []
            k_shards = []
            o_shards = []
            for rank in range(world_size):
                spec = distributor.get_attention_shard_spec(
                    layer_idx=0, rank=rank, world_size=world_size
                )
                assert spec["num_local_attention_heads"] == config.num_attention_heads // world_size
                assert spec["num_local_key_value_heads"] == (config.num_key_value_heads or 0) // world_size
                assert spec["head_dim"] == config.head_dim

                q_w = spec["weights"]["q_proj.weight"]
                k_w = spec["weights"]["k_proj.weight"]
                o_w = spec["weights"]["o_proj.weight"]

                assert q_w.shape == (config.hidden_size // world_size, config.hidden_size)
                assert k_w.shape == (config.kv_hidden_size // world_size, config.hidden_size)
                assert o_w.shape == (config.hidden_size, config.hidden_size // world_size)

                q_shards.append((spec["descriptors"]["q_proj.weight"], q_w))
                k_shards.append((spec["descriptors"]["k_proj.weight"], k_w))
                o_shards.append((spec["descriptors"]["o_proj.weight"], o_w))

            assert np.array_equal(
                WeightDistributor.reconstruct_sharded_tensor(q_shards),
                expected_weights["model.layers.0.self_attn.q_proj.weight"],
            )
            assert np.array_equal(
                WeightDistributor.reconstruct_sharded_tensor(k_shards),
                expected_weights["model.layers.0.self_attn.k_proj.weight"],
            )
            assert np.array_equal(
                WeightDistributor.reconstruct_sharded_tensor(o_shards),
                expected_weights["model.layers.0.self_attn.o_proj.weight"],
            )

    def test_ffn_sharding_for_member_4(
        self,
        single_shard_checkpoint: Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]],
    ) -> None:
        ckpt_dir, config, expected_weights = single_shard_checkpoint
        world_size = 2
        with MemoryMappedWeightStore(ckpt_dir) as store:
            distributor = WeightDistributor(config, store)

            gate_shards = []
            down_shards = []
            for rank in range(world_size):
                spec = distributor.get_ffn_shard_spec(
                    layer_idx=1, rank=rank, world_size=world_size
                )
                assert spec["local_intermediate_size"] == config.intermediate_size // world_size

                gate_w = spec["weights"]["gate_proj.weight"]
                down_w = spec["weights"]["down_proj.weight"]
                assert gate_w.shape == (config.intermediate_size // world_size, config.hidden_size)
                assert down_w.shape == (config.hidden_size, config.intermediate_size // world_size)

                gate_shards.append((spec["descriptors"]["gate_proj.weight"], gate_w))
                down_shards.append((spec["descriptors"]["down_proj.weight"], down_w))

            assert np.array_equal(
                WeightDistributor.reconstruct_sharded_tensor(gate_shards),
                expected_weights["model.layers.1.mlp.gate_proj.weight"],
            )
            assert np.array_equal(
                WeightDistributor.reconstruct_sharded_tensor(down_shards),
                expected_weights["model.layers.1.mlp.down_proj.weight"],
            )

    def test_cluster_distribution_and_safetensors_export_for_member_1(
        self,
        tmp_path: Path,
        single_shard_checkpoint: Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]],
    ) -> None:
        ckpt_dir, config, _ = single_shard_checkpoint
        workers = [
            WorkerShardTarget(node_id=f"worker-{r}", rank=r, world_size=2, port=50051 + r)
            for r in range(2)
        ]

        export_dir = tmp_path / "exported_node_shards"
        with MemoryMappedWeightStore(ckpt_dir) as store:
            distributor = WeightDistributor(config, store)
            packages = distributor.prepare_cluster_distribution(workers)
            assert len(packages) == 2

            for rank, pkg in packages.items():
                assert pkg.total_bytes > 0
                out_file = pkg.export_safetensors(export_dir)
                assert out_file.exists()
                with MemoryMappedWeightStore(out_file) as node_store:
                    q_slice = node_store.get_tensor("model.layers.0.self_attn.q_proj.weight")
                    assert q_slice.shape == (config.hidden_size // 2, config.hidden_size)

    def test_invalid_worker_ranks_raise(
        self,
        single_shard_checkpoint: Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]],
    ) -> None:
        ckpt_dir, config, _ = single_shard_checkpoint
        bad_workers = [
            WorkerShardTarget(node_id="w0", rank=0, world_size=2),
            WorkerShardTarget(node_id="w2", rank=2, world_size=2),
        ]
        with MemoryMappedWeightStore(ckpt_dir) as store:
            distributor = WeightDistributor(config, store)
            with pytest.raises(ShardDistributionError, match="contiguous ranks"):
                distributor.prepare_cluster_distribution(bad_workers)
