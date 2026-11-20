"""
Shared PyTest fixtures generating synthetic Hugging Face LLM checkpoints
(both single-file `.safetensors` and multi-shard `.safetensors.index.json`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from dmg.model_loader.config import LLMModelConfig
from dmg.model_loader.safetensors_io import save_safetensors_file


def build_synthetic_weights(
    config: LLMModelConfig, seed: int = 42
) -> Dict[str, np.ndarray[Any, Any]]:
    """Create deterministic synthetic float32/float16 weights matching `config`."""
    rng = np.random.default_rng(seed)
    np_dtype = np.float16 if config.torch_dtype == "F16" else np.float32
    tensors: Dict[str, np.ndarray[Any, Any]] = {}
    for name, shape in config.expected_tensor_shapes().items():
        arr = rng.standard_normal(size=shape).astype(np_dtype)
        tensors[name] = arr
    return tensors


@pytest.fixture
def tiny_model_config() -> LLMModelConfig:
    """Return a compact, fast-to-test GQA Transformer model configuration."""
    return LLMModelConfig(
        model_name_or_path="dmg-llama-tiny-test",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=512,
        torch_dtype="F32",
        tie_word_embeddings=False,
    )


@pytest.fixture
def single_shard_checkpoint(
    tmp_path: Path, tiny_model_config: LLMModelConfig
) -> Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]]:
    """Create a complete single-file `model.safetensors` + `config.json` checkpoint."""
    ckpt_dir = tmp_path / "single_shard_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tiny_model_config.save_json(ckpt_dir / "config.json")
    weights = build_synthetic_weights(tiny_model_config)
    save_safetensors_file(
        weights,
        ckpt_dir / "model.safetensors",
        metadata={"format": "pt", "creator": "dmg-test-suite"},
    )
    return ckpt_dir, tiny_model_config, weights


@pytest.fixture
def multi_shard_checkpoint(
    tmp_path: Path, tiny_model_config: LLMModelConfig
) -> Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]]:
    """Create a 2-shard checkpoint with `model.safetensors.index.json`."""
    ckpt_dir = tmp_path / "multi_shard_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tiny_model_config.save_json(ckpt_dir / "config.json")
    weights = build_synthetic_weights(tiny_model_config, seed=99)

    names = list(weights.keys())
    midpoint = len(names) // 2
    shard1_tensors = {k: weights[k] for k in names[:midpoint]}
    shard2_tensors = {k: weights[k] for k in names[midpoint:]}

    shard1_name = "model-00001-of-00002.safetensors"
    shard2_name = "model-00002-of-00002.safetensors"
    save_safetensors_file(shard1_tensors, ckpt_dir / shard1_name)
    save_safetensors_file(shard2_tensors, ckpt_dir / shard2_name)

    weight_map: Dict[str, str] = {}
    for k in shard1_tensors:
        weight_map[k] = shard1_name
    for k in shard2_tensors:
        weight_map[k] = shard2_name

    index_payload = {
        "metadata": {"total_size": tiny_model_config.estimate_memory_bytes()},
        "weight_map": weight_map,
    }
    (ckpt_dir / "model.safetensors.index.json").write_text(
        json.dumps(index_payload, indent=2), encoding="utf-8"
    )
    return ckpt_dir, tiny_model_config, weights
