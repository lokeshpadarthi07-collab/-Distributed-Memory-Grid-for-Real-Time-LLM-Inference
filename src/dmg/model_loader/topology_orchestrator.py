"""
Automated Weight Shard Pre-Allocation, Memory-Mapped Pinning, and Topology Validator.

Coordinates with Member 1 (Master-Worker Network) to pre-allocate memory-mapped
shared regions, generate worker-specific deployment packages, and validate
network topology configurations before runtime inference execution.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from dmg.core.exceptions import ConfigurationError, ShardDistributionError
from dmg.core.interfaces import WorkerShardTarget
from dmg.model_loader.config import ConfigValidator, LLMModelConfig
from dmg.model_loader.mmap_loader import MemoryMappedWeightStore
from dmg.model_loader.sharding import NodeShardPackage, PartitionStrategy, WeightDistributor


@dataclass(frozen=True)
class ShardAllocationPlan:
    """Detailed memory allocation blueprint for a specific worker node in the topology."""

    node_id: str
    rank: int
    world_size: int
    assigned_tensors: List[str]
    total_allocated_bytes: int
    column_parallel_count: int
    row_parallel_count: int
    replicated_count: int


@dataclass
class TopologyPreflightReport:
    """Validation report ensuring cluster topology matches model parallel requirements."""

    model_name: str
    world_size: int
    is_valid: bool
    per_node_allocations: Dict[int, ShardAllocationPlan] = field(default_factory=dict)
    validation_messages: List[str] = field(default_factory=list)


class ShardTopologyOrchestrator:
    """
    Orchestrates cluster topology verification and shard memory allocation planning
    before initiating distributed worker processes in Member 1's architecture.
    """

    def __init__(
        self,
        config: LLMModelConfig,
        store: MemoryMappedWeightStore,
    ) -> None:
        self.config = config
        self.store = store
        self.distributor = WeightDistributor(config, store)

    def validate_and_plan_topology(
        self,
        workers: List[WorkerShardTarget],
    ) -> TopologyPreflightReport:
        """
        Verify that `workers` meet tensor parallel requirements and construct
        the complete memory-mapped pre-allocation plan for each node.
        """
        if not workers:
            raise ShardDistributionError("Cannot plan topology with empty worker target list.")

        world_size = len(workers)
        ranks = sorted(w.rank for w in workers)
        messages: List[str] = []

        if ranks != list(range(world_size)):
            raise ShardDistributionError(
                f"Worker ranks must be contiguous 0..{world_size - 1}, got {ranks}"
            )

        # Validate tensor parallelism divisibility
        try:
            ConfigValidator.validate_parallelism(self.config, world_size)
            messages.append(
                f"Cluster world_size={world_size} satisfies attention head and FFN intermediate divisibility."
            )
        except ConfigurationError as e:
            return TopologyPreflightReport(
                model_name=self.config.model_name_or_path,
                world_size=world_size,
                is_valid=False,
                validation_messages=[str(e)],
            )

        plans: Dict[int, ShardAllocationPlan] = {}
        all_tensors = self.store.list_tensors()

        for worker in workers:
            col_cnt = 0
            row_cnt = 0
            rep_cnt = 0
            node_bytes = 0
            assigned: List[str] = []

            for name in all_tensors:
                meta = self.store.get_tensor_metadata(name)
                strategy, split_dim = WeightDistributor.classify_tensor_strategy(name)

                if strategy == PartitionStrategy.COLUMN_PARALLEL:
                    col_cnt += 1
                    node_bytes += meta.num_bytes // world_size
                elif strategy == PartitionStrategy.ROW_PARALLEL:
                    row_cnt += 1
                    node_bytes += meta.num_bytes // world_size
                else:
                    rep_cnt += 1
                    node_bytes += meta.num_bytes

                assigned.append(name)

            plans[worker.rank] = ShardAllocationPlan(
                node_id=worker.node_id,
                rank=worker.rank,
                world_size=world_size,
                assigned_tensors=assigned,
                total_allocated_bytes=node_bytes,
                column_parallel_count=col_cnt,
                row_parallel_count=row_cnt,
                replicated_count=rep_cnt,
            )

        return TopologyPreflightReport(
            model_name=self.config.model_name_or_path,
            world_size=world_size,
            is_valid=True,
            per_node_allocations=plans,
            validation_messages=messages,
        )

    def export_topology_blueprint(
        self,
        report: TopologyPreflightReport,
        output_path: Union[str, Path],
    ) -> Path:
        """Serialize topology preflight report to JSON for Member 1 master node orchestration."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(report)
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return out
