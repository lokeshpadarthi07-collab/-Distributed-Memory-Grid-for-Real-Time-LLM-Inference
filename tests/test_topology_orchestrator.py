"""
Unit tests for ShardTopologyOrchestrator and preflight validation (`topology_orchestrator.py`).
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
from dmg.model_loader.topology_orchestrator import ShardTopologyOrchestrator


def test_topology_orchestrator_planning(
    tmp_path: Path,
    single_shard_checkpoint: Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]],
) -> None:
    ckpt_dir, config, weights = single_shard_checkpoint

    workers = [
        WorkerShardTarget(node_id=f"worker-{r}", rank=r, world_size=4)
        for r in range(4)
    ]

    with MemoryMappedWeightStore(ckpt_dir) as store:
        orchestrator = ShardTopologyOrchestrator(config, store)
        report = orchestrator.validate_and_plan_topology(workers)

        assert report.is_valid
        assert report.world_size == 4
        assert len(report.per_node_allocations) == 4

        plan0 = report.per_node_allocations[0]
        assert plan0.node_id == "worker-0"
        assert plan0.column_parallel_count > 0
        assert plan0.row_parallel_count > 0
        assert plan0.total_allocated_bytes < config.estimate_memory_bytes()

        # Test JSON blueprint export
        out_file = tmp_path / "topology_blueprint.json"
        saved = orchestrator.export_topology_blueprint(report, out_file)
        assert saved.exists()


def test_topology_orchestrator_invalid_ranks(
    single_shard_checkpoint: Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]],
) -> None:
    ckpt_dir, config, weights = single_shard_checkpoint
    bad_workers = [
        WorkerShardTarget(node_id="worker-0", rank=0, world_size=2),
        WorkerShardTarget(node_id="worker-2", rank=2, world_size=2),
    ]

    with MemoryMappedWeightStore(ckpt_dir) as store:
        orchestrator = ShardTopologyOrchestrator(config, store)
        with pytest.raises(ShardDistributionError, match="contiguous"):
            orchestrator.validate_and_plan_topology(bad_workers)
