"""
Unit tests for MemoryMappedPageWarmer and OS page-fault warmup (`page_warmer.py`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from dmg.model_loader.config import LLMModelConfig
from dmg.model_loader.mmap_loader import MemoryMappedWeightStore
from dmg.model_loader.page_warmer import MemoryMappedPageWarmer


def test_mmap_page_warmer_execution(
    single_shard_checkpoint: Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]]
) -> None:
    ckpt_dir, config, weights = single_shard_checkpoint

    with MemoryMappedWeightStore(ckpt_dir) as store:
        warmer = MemoryMappedPageWarmer(store, max_workers=2)

        # Warm up subset of tensors
        target_tensors = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ]
        summary = warmer.warmup_tensors(target_tensors)

        assert summary.total_tensors_warmed == 2
        assert summary.total_bytes_warmed > 0
        assert summary.total_pages_touched > 0
        assert summary.throughput_mb_per_sec > 0.0
        assert len(summary.results) == 2

        # Warm up entire store
        full_summary = warmer.warmup_tensors()
        assert full_summary.total_tensors_warmed == len(weights)
        assert full_summary.total_bytes_warmed == sum(w.nbytes for w in weights.values())
