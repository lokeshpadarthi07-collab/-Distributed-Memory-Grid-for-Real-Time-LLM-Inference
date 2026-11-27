"""
Model configuration, architecture specification, and validation for LLM
weight loading in the Distributed Memory Grid.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from dmg.core.exceptions import ConfigurationError

SUPPORTED_DTYPES: Dict[str, Tuple[str, int]] = {
    "F32": ("float32", 4),
    "float32": ("float32", 4),
    "fp32": ("float32", 4),
    "F16": ("float16", 2),
    "float16": ("float16", 2),
    "fp16": ("float16", 2),
    "BF16": ("bfloat16", 2),
    "bfloat16": ("bfloat16", 2),
    "bf16": ("bfloat16", 2),
    "I64": ("int64", 8),
    "int64": ("int64", 8),
    "I32": ("int32", 4),
    "int32": ("int32", 4),
    "I8": ("int8", 1),
    "int8": ("int8", 1),
    "U8": ("uint8", 1),
    "uint8": ("uint8", 1),
}

NP_TO_SAFETENSORS_DTYPE: Dict[str, str] = {
    "float32": "F32",
    "float16": "F16",
    "bfloat16": "BF16",
    "int64": "I64",
    "int32": "I32",
    "int8": "I8",
    "uint8": "U8",
}


def normalize_dtype(dtype_str: str) -> str:
    """Normalize a dtype identifier to canonical Safetensors format (e.g. F32, F16, BF16)."""
    if dtype_str not in SUPPORTED_DTYPES:
        raise ConfigurationError(
            f"Unsupported tensor dtype '{dtype_str}'. Supported: {sorted(SUPPORTED_DTYPES.keys())}"
        )
    canonical_name, _ = SUPPORTED_DTYPES[dtype_str]
    return NP_TO_SAFETENSORS_DTYPE[canonical_name]


def dtype_byte_size(dtype_str: str) -> int:
    """Return byte width per scalar element for a supported dtype."""
    if dtype_str not in SUPPORTED_DTYPES:
        raise ConfigurationError(f"Unsupported tensor dtype '{dtype_str}'.")
    return SUPPORTED_DTYPES[dtype_str][1]


@dataclass
class LLMModelConfig:
    """
    Configuration dataclass representing a Transformer LLM architecture
    (compatible with LLaMA / Mistral / Qwen style Hugging Face `config.json`).
    """

    model_name_or_path: str = "dmg-llama-tiny"
    architectures: Tuple[str, ...] = ("LlamaForCausalLM",)
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    torch_dtype: str = "F32"
    attention_bias: bool = False
    mlp_bias: bool = False
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.torch_dtype = normalize_dtype(self.torch_dtype)
        if isinstance(self.architectures, list):
            self.architectures = tuple(self.architectures)

    @property
    def head_dim(self) -> int:
        """Dimension of each individual attention head."""
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_hidden_size(self) -> int:
        """Total projection dimension for Key and Value tensors (supports GQA/MQA)."""
        assert self.num_key_value_heads is not None
        return self.head_dim * self.num_key_value_heads

    @property
    def num_queries_per_kv(self) -> int:
        """Number of query heads sharing a single KV head in Grouped-Query Attention."""
        assert self.num_key_value_heads is not None
        return self.num_attention_heads // self.num_key_value_heads

    def expected_tensor_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """
        Compute the canonical map of parameter names to expected tensor shapes
        for the configured Transformer architecture.
        """
        shapes: Dict[str, Tuple[int, ...]] = {
            "model.embed_tokens.weight": (self.vocab_size, self.hidden_size),
            "model.norm.weight": (self.hidden_size,),
        }
        if not self.tie_word_embeddings:
            shapes["lm_head.weight"] = (self.vocab_size, self.hidden_size)

        for layer_idx in range(self.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            # Self-attention projections (Member 3 target)
            shapes[f"{prefix}.self_attn.q_proj.weight"] = (
                self.hidden_size,
                self.hidden_size,
            )
            shapes[f"{prefix}.self_attn.k_proj.weight"] = (
                self.kv_hidden_size,
                self.hidden_size,
            )
            shapes[f"{prefix}.self_attn.v_proj.weight"] = (
                self.kv_hidden_size,
                self.hidden_size,
            )
            shapes[f"{prefix}.self_attn.o_proj.weight"] = (
                self.hidden_size,
                self.hidden_size,
            )
            if self.attention_bias:
                shapes[f"{prefix}.self_attn.q_proj.bias"] = (self.hidden_size,)
                shapes[f"{prefix}.self_attn.k_proj.bias"] = (self.kv_hidden_size,)
                shapes[f"{prefix}.self_attn.v_proj.bias"] = (self.kv_hidden_size,)
                shapes[f"{prefix}.self_attn.o_proj.bias"] = (self.hidden_size,)

            # Feed-forward projections (Member 4 target)
            shapes[f"{prefix}.mlp.gate_proj.weight"] = (
                self.intermediate_size,
                self.hidden_size,
            )
            shapes[f"{prefix}.mlp.up_proj.weight"] = (
                self.intermediate_size,
                self.hidden_size,
            )
            shapes[f"{prefix}.mlp.down_proj.weight"] = (
                self.hidden_size,
                self.intermediate_size,
            )
            if self.mlp_bias:
                shapes[f"{prefix}.mlp.gate_proj.bias"] = (self.intermediate_size,)
                shapes[f"{prefix}.mlp.up_proj.bias"] = (self.intermediate_size,)
                shapes[f"{prefix}.mlp.down_proj.bias"] = (self.hidden_size,)

            # Layer norms
            shapes[f"{prefix}.input_layernorm.weight"] = (self.hidden_size,)
            shapes[f"{prefix}.post_attention_layernorm.weight"] = (self.hidden_size,)

        return shapes

    def estimate_parameter_count(self) -> int:
        """Return total number of scalar parameters across the model."""
        total = 0
        for shape in self.expected_tensor_shapes().values():
            prod = 1
            for dim in shape:
                prod *= dim
            total += prod
        return total

    def estimate_memory_bytes(self) -> int:
        """Return estimated memory footprint in bytes for all model weights."""
        return self.estimate_parameter_count() * dtype_byte_size(self.torch_dtype)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a JSON-compatible dictionary."""
        data = asdict(self)
        data["architectures"] = list(self.architectures)
        return data

    def save_json(self, path: Union[str, Path]) -> Path:
        """Write configuration to a Hugging Face compatible `config.json` file."""
        out_path = Path(path)
        if out_path.is_dir():
            out_path = out_path / "config.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out_path

    @classmethod
    def from_dict(
        cls, raw: Dict[str, Any], model_name_or_path: str = "dmg-model"
    ) -> "LLMModelConfig":
        """Construct an `LLMModelConfig` from a dictionary (e.g., parsed `config.json`)."""
        known_fields = {
            "model_name_or_path",
            "architectures",
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "rms_norm_eps",
            "rope_theta",
            "tie_word_embeddings",
            "torch_dtype",
            "attention_bias",
            "mlp_bias",
        }
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for k, v in raw.items():
            if k in known_fields:
                kwargs[k] = v
            elif k != "extra_metadata":
                extra[k] = v
        kwargs.setdefault("model_name_or_path", model_name_or_path)
        kwargs["extra_metadata"] = {** raw.get("extra_metadata", {}), **extra}
        config = cls(**kwargs)
        ConfigValidator.validate(config)
        return config

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "LLMModelConfig":
        """Load and validate an `LLMModelConfig` from a file or directory."""
        target = Path(path)
        if target.is_dir():
            target = target / "config.json"
        if not target.exists():
            raise ConfigurationError(f"Model configuration file not found: {target}")
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                f"Invalid JSON in configuration file {target}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise ConfigurationError(
                f"Expected JSON object in {target}, got {type(raw).__name__}"
            )
        return cls.from_dict(raw, model_name_or_path=str(target.parent))


class ConfigValidator:
    """
    Validates structural, mathematical, and distributed-parallelism constraints
    on an `LLMModelConfig`.
    """

    @staticmethod
    def validate(config: LLMModelConfig, world_size: Optional[int] = None) -> None:
        """
        Validate architectural invariants and optional tensor-parallel `world_size`
        compatibility. Raises `ConfigurationError` on any violation.
        """
        if config.vocab_size <= 0:
            raise ConfigurationError(
                f"vocab_size must be positive, got {config.vocab_size}"
            )
        if config.hidden_size <= 0:
            raise ConfigurationError(
                f"hidden_size must be positive, got {config.hidden_size}"
            )
        if config.intermediate_size <= 0:
            raise ConfigurationError(
                f"intermediate_size must be positive, got {config.intermediate_size}"
            )
        if config.num_hidden_layers <= 0:
            raise ConfigurationError(
                f"num_hidden_layers must be positive, got {config.num_hidden_layers}"
            )
        if config.num_attention_heads <= 0:
            raise ConfigurationError(
                f"num_attention_heads must be positive, got {config.num_attention_heads}"
            )
        if config.num_key_value_heads is None or config.num_key_value_heads <= 0:
            raise ConfigurationError(
                f"num_key_value_heads must be positive, got {config.num_key_value_heads}"
            )

        if config.hidden_size % config.num_attention_heads != 0:
            raise ConfigurationError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )

        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ConfigurationError(
                f"num_attention_heads ({config.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({config.num_key_value_heads}) for GQA/MQA compatibility"
            )

        if config.head_dim % 2 != 0:
            raise ConfigurationError(
                f"head_dim ({config.head_dim}) must be an even integer for Rotary Position Embeddings (RoPE)"
            )

        if config.max_position_embeddings <= 0:
            raise ConfigurationError(
                f"max_position_embeddings must be positive, got {config.max_position_embeddings}"
            )

        if config.rms_norm_eps <= 0:
            raise ConfigurationError(
                f"rms_norm_eps must be > 0, got {config.rms_norm_eps}"
            )

        if world_size is not None:
            ConfigValidator.validate_parallelism(config, world_size)

    @staticmethod
    def validate_parallelism(config: LLMModelConfig, world_size: int) -> None:
        """
        Validate that the model's attention heads (Member 3) and FFN dimensions (Member 4)
        can be evenly partitioned across `world_size` worker nodes.
        """
        if world_size <= 0:
            raise ConfigurationError(f"world_size must be >= 1, got {world_size}")

        assert config.num_key_value_heads is not None
        if config.num_attention_heads % world_size != 0:
            raise ConfigurationError(
                f"num_attention_heads ({config.num_attention_heads}) is not evenly divisible "
                f"by world_size ({world_size})"
            )

        if config.num_key_value_heads % world_size != 0:
            raise ConfigurationError(
                f"num_key_value_heads ({config.num_key_value_heads}) is not evenly divisible "
                f"by world_size ({world_size})"
            )

        if config.intermediate_size % world_size != 0:
            raise ConfigurationError(
                f"intermediate_size ({config.intermediate_size}) is not evenly divisible "
                f"by world_size ({world_size})"
            )
