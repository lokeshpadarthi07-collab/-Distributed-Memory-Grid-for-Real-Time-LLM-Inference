"""
Asynchronous tensor streaming loader and memory-mapped batch prefetch engine.

Provides non-blocking, asynchronous loading of Safetensors weights to feed
Member 1's gRPC streaming transport and Member 5's memory pool with zero pipeline stalls.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import numpy as np

from dmg.core.interfaces import TensorMetadata
from dmg.model_loader.loader import LLMWeightLoader, TransformerLayerWeights


@dataclass(frozen=True)
class StreamedTensorChunk:
    """A sliced payload chunk of a model tensor emitted over async streams."""

    tensor_name: str
    chunk_index: int
    total_chunks: int
    data: np.ndarray[Any, Any]
    is_last: bool


class AsyncTensorStreamer:
    """
    Asynchronous streaming wrapper over `LLMWeightLoader`.
    Enables async/await tensor streaming, asynchronous layer pipelines,
    and non-blocking prefetching into memory mapped regions.
    """

    def __init__(
        self,
        loader: LLMWeightLoader,
        *,
        chunk_size_elements: int = 1_000_000,
        executor_threads: int = 4,
    ) -> None:
        self.loader = loader
        self.chunk_size_elements = chunk_size_elements
        self._executor = ThreadPoolExecutor(
            max_workers=executor_threads,
            thread_name_prefix="dmg-async-streamer",
        )

    async def get_tensor_async(
        self, tensor_name: str, *, copy: bool = False
    ) -> np.ndarray[Any, Any]:
        """Asynchronously load a tensor by offloading memory mapping to a worker thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.loader.load_tensor(tensor_name, copy=copy),
        )

    async def get_layer_async(
        self, layer_idx: int, *, copy: bool = False
    ) -> TransformerLayerWeights:
        """Asynchronously load all weights for a transformer layer block."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.loader.load_layer(layer_idx, copy=copy),
        )

    async def stream_tensor_chunks(
        self, tensor_name: str
    ) -> AsyncIterator[StreamedTensorChunk]:
        """
        Stream a large parameter tensor in discrete chunked slices
        compatible with gRPC chunk streaming (Member 1) or incremental caching (Member 5).
        """
        arr = await self.get_tensor_async(tensor_name, copy=False)
        flat = arr.reshape(-1)
        total_elements = flat.shape[0]

        if total_elements == 0:
            yield StreamedTensorChunk(
                tensor_name=tensor_name,
                chunk_index=0,
                total_chunks=1,
                data=arr,
                is_last=True,
            )
            return

        total_chunks = (total_elements + self.chunk_size_elements - 1) // self.chunk_size_elements

        for idx in range(total_chunks):
            start = idx * self.chunk_size_elements
            end = min(total_elements, start + self.chunk_size_elements)
            chunk_slice = flat[start:end]
            is_last = (idx == total_chunks - 1)
            yield StreamedTensorChunk(
                tensor_name=tensor_name,
                chunk_index=idx,
                total_chunks=total_chunks,
                data=chunk_slice,
                is_last=is_last,
            )
            # Yield control back to asyncio event loop
            await asyncio.sleep(0)

    async def stream_layers_pipeline(
        self,
        layer_indices: Optional[List[int]] = None,
        *,
        prefetch_window: int = 2,
    ) -> AsyncIterator[Tuple[int, TransformerLayerWeights]]:
        """
        Asynchronously stream transformer layers while maintaining a sliding
        prefetch window to overlap disk read/decompression with inference compute.
        """
        num_layers = self.loader.config.num_hidden_layers
        indices = layer_indices if layer_indices is not None else list(range(num_layers))

        loop = asyncio.get_running_loop()

        # Prefetch initial window
        self.loader.prefetch_layers(indices[:prefetch_window])

        for pos, layer_idx in enumerate(indices):
            # Schedule lookahead layer
            lookahead_pos = pos + prefetch_window
            if lookahead_pos < len(indices):
                self.loader.prefetch_layers([indices[lookahead_pos]])

            layer_weights = await self.get_layer_async(layer_idx, copy=False)
            yield (layer_idx, layer_weights)

    def close(self) -> None:
        """Shutdown the streaming threadpool executor."""
        self._executor.shutdown(wait=False)

    async def __aenter__(self) -> "AsyncTensorStreamer":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
