"""
Unit tests for Week 1:
- Model configuration loading, serialization, and architectural validation (`config.py`)
- Hugging Face Safetensors binary header parsing and multi-shard indexing (`safetensors_io.py`)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from dmg.core.exceptions import ConfigurationError, SafetensorsFormatError
from dmg.model_loader.config import (
    ConfigValidator,
    LLMModelConfig,
    dtype_byte_size,
    normalize_dtype,
)
from dmg.model_loader.safetensors_io import (
    SafetensorsIndex,
    parse_safetensors_header,
    save_safetensors_file,
)


class TestModelConfiguration:
    """Tests for `LLMModelConfig` and `ConfigValidator`."""

    def test_valid_gqa_configuration(self, tiny_model_config: LLMModelConfig) -> None:
        ConfigValidator.validate(tiny_model_config, world_size=4)
        assert tiny_model_config.head_dim == 8
        assert tiny_model_config.kv_hidden_size == 32
        assert tiny_model_config.num_queries_per_kv == 2
        assert tiny_model_config.estimate_parameter_count() > 0
        assert (
            tiny_model_config.estimate_memory_bytes()
            == tiny_model_config.estimate_parameter_count() * 4
        )

    def test_config_json_roundtrip(
        self, tmp_path: Path, tiny_model_config: LLMModelConfig
    ) -> None:
        json_path = tiny_model_config.save_json(tmp_path / "config.json")
        loaded = LLMModelConfig.from_json(json_path)
        assert loaded.vocab_size == tiny_model_config.vocab_size
        assert loaded.hidden_size == tiny_model_config.hidden_size
        assert loaded.num_attention_heads == tiny_model_config.num_attention_heads
        assert loaded.num_key_value_heads == tiny_model_config.num_key_value_heads

    def test_invalid_head_divisibility_raises(self) -> None:
        cfg = LLMModelConfig(hidden_size=64, num_attention_heads=6)
        with pytest.raises(ConfigurationError, match="divisible by num_attention_heads"):
            ConfigValidator.validate(cfg)

    def test_invalid_gqa_ratio_raises(self) -> None:
        cfg = LLMModelConfig(
            hidden_size=64, num_attention_heads=8, num_key_value_heads=3
        )
        with pytest.raises(ConfigurationError, match="divisible by num_key_value_heads"):
            ConfigValidator.validate(cfg)

    def test_invalid_parallelism_world_size_raises(
        self, tiny_model_config: LLMModelConfig
    ) -> None:
        # num_key_value_heads=4 cannot be divided evenly across 8 ranks
        with pytest.raises(ConfigurationError, match="not evenly divisible by world_size"):
            ConfigValidator.validate(tiny_model_config, world_size=8)

    def test_dtype_normalization(self) -> None:
        assert normalize_dtype("float16") == "F16"
        assert normalize_dtype("bf16") == "BF16"
        assert normalize_dtype("F32") == "F32"
        assert dtype_byte_size("F16") == 2
        with pytest.raises(ConfigurationError):
            normalize_dtype("complex128")


class TestSafetensorsIO:
    """Tests for `parse_safetensors_header` and `SafetensorsIndex`."""

    def test_parse_header_without_loading_weights(
        self,
        single_shard_checkpoint: Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]],
    ) -> None:
        ckpt_dir, config, weights = single_shard_checkpoint
        st_file = ckpt_dir / "model.safetensors"
        header = parse_safetensors_header(st_file)

        assert header.file_path == st_file.resolve()
        assert header.data_base_offset == 8 + header.header_size
        assert len(header.tensors) == len(weights)
        assert header.custom_metadata.get("creator") == "dmg-test-suite"

        embed_meta = header.tensors["model.embed_tokens.weight"]
        assert embed_meta.shape == (config.vocab_size, config.hidden_size)
        assert embed_meta.dtype == "F32"
        assert embed_meta.num_bytes == config.vocab_size * config.hidden_size * 4

    def test_multi_shard_index_resolution(
        self,
        multi_shard_checkpoint: Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]],
    ) -> None:
        ckpt_dir, config, weights = multi_shard_checkpoint
        index = SafetensorsIndex(ckpt_dir)

        assert index.num_shards == 2
        assert set(index.list_tensors()) == set(weights.keys())
        meta = index.get_metadata("model.layers.1.mlp.down_proj.weight")
        assert meta.shape == (config.hidden_size, config.intermediate_size)

    def test_corrupted_safetensors_file_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "corrupt.safetensors"
        bad_file.write_bytes(b"short")
        with pytest.raises(SafetensorsFormatError, match="too small"):
            parse_safetensors_header(bad_file)
