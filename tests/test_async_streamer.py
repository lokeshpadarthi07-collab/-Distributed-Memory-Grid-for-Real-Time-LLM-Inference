"""
Unit tests for Asynchronous Tensor Streaming and Pipeline Prefetching (`async_streamer.py`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from dmg.model_loader.async_streamer import AsyncTensorStreamer
from dmg.model_loader.config import LLMModelConfig
from dmg.model_loader.loader import LLMWeightLoader


@pytest.mark.asyncio
async def test_async_tensor_streaming_and_reconstruction(
    single_shard_checkpoint: Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]]
) -> None:
    ckpt_dir, config, weights = single_shard_checkpoint

    with LLMWeightLoader(ckpt_dir) as loader:
        streamer = AsyncTensorStreamer(loader, chunk_size_elements=64)
        target_tensor = "model.embed_tokens.weight"

        # Test single tensor async load
        loaded_arr = await streamer.get_tensor_async(target_tensor)
        assert np.array_equal(loaded_arr, weights[target_tensor])

        # Test chunked async stream
        chunks = []
        async for chunk in streamer.stream_tensor_chunks(target_tensor):
            chunks.append(chunk.data)

        reconstructed_flat = np.concatenate(chunks)
        assert np.array_equal(reconstructed_flat, weights[target_tensor].reshape(-1))
        streamer.close()


@pytest.mark.asyncio
async def test_async_layers_pipeline_streaming(
    single_shard_checkpoint: Tuple[Path, LLMModelConfig, Dict[str, np.ndarray[Any, Any]]]
) -> None:
    ckpt_dir, config, weights = single_shard_checkpoint

    with LLMWeightLoader(ckpt_dir) as loader:
        async with AsyncTensorStreamer(loader) as streamer:
            streamed_layers = []
            async for layer_idx, layer in streamer.stream_layers_pipeline(prefetch_window=1):
                streamed_layers.append(layer_idx)
                assert layer.q_proj.shape == weights[f"model.layers.{layer_idx}.self_attn.q_proj.weight"].shape

            assert streamed_layers == list(range(config.num_hidden_layers))
