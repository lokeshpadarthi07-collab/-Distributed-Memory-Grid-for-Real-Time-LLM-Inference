"""
Foundational LLM Model Weight Loader (Week 1).

Coordinates `LLMModelConfig` and `MemoryMappedWeightStore` to provide
structured layer-by-layer and component-level access to Transformer weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from dmg.core.exceptions import ConfigurationError
from dmg.core.interfaces import TensorMetadata, WeightLoaderProtocol
from dmg.model_loader.config import ConfigValidator, LLMModelConfig
from dmg.model_loader.mmap_loader import MemoryMappedWeightStore


@dataclass
class TransformerLayerWeights:
    """Container holding all memory-mapped parameter arrays for one Transformer block."""

    layer_idx: int
    input_layernorm: np.ndarray[Any, Any]
    post_attention_layernorm: np.ndarray[Any, Any]
    q_proj: np.ndarray[Any, Any]
    k_proj: np.ndarray[Any, Any]
    v_proj: np.ndarray[Any, Any]
    o_proj: np.ndarray[Any, Any]
    gate_proj: np.ndarray[Any, Any]
    up_proj: np.ndarray[Any, Any]
    down_proj: np.ndarray[Any, Any]


class LLMWeightLoader(WeightLoaderProtocol):
    """
    Foundational LLM weight loader backed by zero-copy memory-mapped Safetensors.
    Implements `WeightLoaderProtocol` for seamless integration across the
    Distributed Memory Grid.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        config: Optional[LLMModelConfig] = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        if config is not None:
            self.config = config
            ConfigValidator.validate(self.config)
        else:
            self.config = LLMModelConfig.from_json(self.checkpoint_dir)

        self.store = MemoryMappedWeightStore(self.checkpoint_dir)

    def get_tensor_metadata(self, tensor_name: str) -> TensorMetadata:
        """Return metadata for a specific model parameter tensor."""
        return self.store.get_tensor_metadata(tensor_name)

    def load_tensor(
        self,
        tensor_name: str,
        *,
        copy: bool = False,
    ) -> np.ndarray[Any, Any]:
        """Load a parameter tensor by name via memory mapping."""
        return self.store.get_tensor(tensor_name, copy=copy)

    def list_tensors(self) -> List[str]:
        """List all tensor names in the checkpoint."""
        return self.store.list_tensors()

    def load_embeddings(self, *, copy: bool = False) -> Dict[str, np.ndarray[Any, Any]]:
        """Load input token embeddings, final RMSNorm, and output LM head weights."""
        result: Dict[str, np.ndarray[Any, Any]] = {
            "embed_tokens": self.load_tensor("model.embed_tokens.weight", copy=copy),
            "norm": self.load_tensor("model.norm.weight", copy=copy),
        }
        if self.config.tie_word_embeddings:
            result["lm_head"] = result["embed_tokens"]
        else:
            result["lm_head"] = self.load_tensor("lm_head.weight", copy=copy)
        return result

    def load_layer(
        self,
        layer_idx: int,
        *,
        copy: bool = False,
    ) -> TransformerLayerWeights:
        """Load all attention, MLP, and normalization weights for `layer_idx`."""
        if layer_idx < 0 or layer_idx >= self.config.num_hidden_layers:
            raise ConfigurationError(
                f"layer_idx {layer_idx} out of range [0, {self.config.num_hidden_layers})"
            )

        prefix = f"model.layers.{layer_idx}"
        return TransformerLayerWeights(
            layer_idx=layer_idx,
            input_layernorm=self.load_tensor(f"{prefix}.input_layernorm.weight", copy=copy),
            post_attention_layernorm=self.load_tensor(
                f"{prefix}.post_attention_layernorm.weight", copy=copy
            ),
            q_proj=self.load_tensor(f"{prefix}.self_attn.q_proj.weight", copy=copy),
            k_proj=self.load_tensor(f"{prefix}.self_attn.k_proj.weight", copy=copy),
            v_proj=self.load_tensor(f"{prefix}.self_attn.v_proj.weight", copy=copy),
            o_proj=self.load_tensor(f"{prefix}.self_attn.o_proj.weight", copy=copy),
            gate_proj=self.load_tensor(f"{prefix}.mlp.gate_proj.weight", copy=copy),
            up_proj=self.load_tensor(f"{prefix}.mlp.up_proj.weight", copy=copy),
            down_proj=self.load_tensor(f"{prefix}.mlp.down_proj.weight", copy=copy),
        )

    def iter_layers(self, *, copy: bool = False) -> Iterator[TransformerLayerWeights]:
        """Lazily yield `TransformerLayerWeights` for each Transformer layer in order."""
        for idx in range(self.config.num_hidden_layers):
            yield self.load_layer(idx, copy=copy)

    def close(self) -> None:
        """Close underlying memory-mapped file handles."""
        self.store.close()

    def __enter__(self) -> "LLMWeightLoader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
