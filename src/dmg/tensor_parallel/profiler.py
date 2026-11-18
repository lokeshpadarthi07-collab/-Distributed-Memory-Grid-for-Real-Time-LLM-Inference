"""
Profiling and latency benchmarking module for Tensor Parallelism.
Measures computation vs communication overhead and speedup across worker nodes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProfilingRecord:
    """Latency breakdown for a single attention forward execution step."""

    step_id: str
    rank: int
    world_size: int
    qkv_projection_ms: float
    attention_compute_ms: float
    out_projection_ms: float
    communication_sync_ms: float
    total_latency_ms: float

    @property
    def compute_ms(self) -> float:
        return (
            self.qkv_projection_ms
            + self.attention_compute_ms
            + self.out_projection_ms
        )

    @property
    def comm_ratio(self) -> float:
        comp = self.compute_ms
        return self.communication_sync_ms / comp if comp > 0 else 0.0


class TensorParallelProfiler:
    """Collects and reports execution metrics across tensor parallel steps."""

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.records: List[ProfilingRecord] = []

    def record_step(
        self,
        step_id: str,
        qkv_ms: float,
        attn_ms: float,
        out_ms: float,
        sync_ms: float,
    ) -> ProfilingRecord:
        """Register a single forward step benchmark measurement."""
        total = qkv_ms + attn_ms + out_ms + sync_ms
        record = ProfilingRecord(
            step_id=step_id,
            rank=self.rank,
            world_size=self.world_size,
            qkv_projection_ms=round(qkv_ms, 4),
            attention_compute_ms=round(attn_ms, 4),
            out_projection_ms=round(out_ms, 4),
            communication_sync_ms=round(sync_ms, 4),
            total_latency_ms=round(total, 4),
        )
        self.records.append(record)
        return record

    def get_summary(self) -> Dict[str, Any]:
        """Calculate aggregated latency averages and communication efficiency."""
        if not self.records:
            return {"status": "no_data", "total_steps": 0}

        n = len(self.records)
        avg_qkv = sum(r.qkv_projection_ms for r in self.records) / n
        avg_attn = sum(r.attention_compute_ms for r in self.records) / n
        avg_out = sum(r.out_projection_ms for r in self.records) / n
        avg_sync = sum(r.communication_sync_ms for r in self.records) / n
        avg_total = sum(r.total_latency_ms for r in self.records) / n
        avg_compute = avg_qkv + avg_attn + avg_out
        comm_overhead_pct = (avg_sync / avg_total * 100) if avg_total > 0 else 0.0

        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "total_steps": n,
            "avg_qkv_ms": round(avg_qkv, 3),
            "avg_attn_ms": round(avg_attn, 3),
            "avg_out_ms": round(avg_out, 3),
            "avg_compute_ms": round(avg_compute, 3),
            "avg_sync_ms": round(avg_sync, 3),
            "avg_total_ms": round(avg_total, 3),
            "comm_overhead_percent": round(comm_overhead_pct, 2),
        }
