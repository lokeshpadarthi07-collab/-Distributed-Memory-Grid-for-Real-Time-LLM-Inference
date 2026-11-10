"""
Cross-module interfaces and data contracts connecting Member 2 (Model Loader)
with Member 1 (Master-Worker Network), Member 3 (Attention Tensor Parallelism),
Member 4 (FFN Distribution), Member 5 (KV-Cache), and Member 6 (API Gateway).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Tuple, runtime_checkable


@dataclass(frozen=True)
class TensorMetadata:
    """Metadata describing a single tensor inside a Safetensors file or shard."""

    name: str
    shape: Tuple[int, ...]
    dtype: str
    data_offsets: Tuple[int, int]
    file_path: str = ""

    @property
    def num_bytes(self) -> int:
        """Total byte length of the tensor payload."""
        return self.data_offsets[1] - self.data_offsets[0]

    @property
    def num_elements(self) -> int:
        """Total number of scalar elements in the tensor."""
        prod = 1
        for dim in self.shape:
            prod *= dim
        return prod


@dataclass(frozen=True)
class WorkerShardTarget:
    """Identifies a target worker node in Member 1's master-worker topology."""

    node_id: str
    rank: int
    world_size: int
    host: str = "127.0.0.1"
    port: int = 50051
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class WeightLoaderProtocol(Protocol):
    """Contract exposed by Member 2 to the rest of the Distributed Memory Grid."""

    def get_tensor_metadata(self, tensor_name: str) -> TensorMetadata:
        """Return metadata for a specific model parameter tensor."""
        ...

    def load_tensor(self, tensor_name: str) -> Any:
        """Load a specific parameter tensor using memory mapping."""
        ...

    def list_tensors(self) -> List[str]:
        """List all available tensor names in the model checkpoint."""
        ...


@runtime_checkable
class AttentionParallelismProvider(Protocol):
    """Contract between Member 2 (weight sharding) and Member 3 (Attention Heads)."""

    def get_attention_shard_spec(
        self, layer_idx: int, rank: int, world_size: int
    ) -> Dict[str, Any]:
        """Return head-split attention weight slices for a given worker rank."""
        ...


@runtime_checkable
class FFNParallelismProvider(Protocol):
    """Contract between Member 2 (weight sharding) and Member 4 (Feed-Forward)."""

    def get_ffn_shard_spec(
        self, layer_idx: int, rank: int, world_size: int
    ) -> Dict[str, Any]:
        """Return column/row-split FFN weight slices for a given worker rank."""
        ...
