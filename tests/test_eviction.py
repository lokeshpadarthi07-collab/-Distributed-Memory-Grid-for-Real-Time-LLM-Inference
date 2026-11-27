"""
Unit tests for DynamicLayerEvictionManager and high-watermark memory budgets (`eviction.py`).
"""

from __future__ import annotations

from dmg.model_loader.eviction import (
    DynamicLayerEvictionManager,
    EvictionPolicy,
)


def test_eviction_manager_under_pressure_and_budget() -> None:
    # 1000 bytes budget: high watermark = 850 bytes, low watermark = 700 bytes
    manager = DynamicLayerEvictionManager(
        max_budget_bytes=1000,
        high_watermark_ratio=0.85,
        low_watermark_ratio=0.70,
        policy=EvictionPolicy.PRIORITY_AWARE,
    )

    manager.register_tensor_access("model.embed_tokens.weight", 300, timestamp=10.0)
    manager.register_tensor_access("model.layers.0.mlp.down_proj.weight", 300, timestamp=11.0)
    manager.register_tensor_access("model.layers.1.mlp.down_proj.weight", 300, timestamp=12.0)

    # Total allocated = 900 bytes > 850 bytes (under pressure)
    assert manager.is_under_pressure()

    # Embed tokens is critical priority 100; layers 0 & 1 MLP are priority 10
    result = manager.evict_to_budget()

    assert result.freed_bytes >= 200
    assert "model.layers.0.mlp.down_proj.weight" in result.evicted_tensors
    assert "model.embed_tokens.weight" not in result.evicted_tensors
    assert manager.current_allocated_bytes <= 700


def test_pinning_prevents_eviction() -> None:
    manager = DynamicLayerEvictionManager(
        max_budget_bytes=500,
        high_watermark_ratio=0.8,
        low_watermark_ratio=0.5,
    )

    manager.register_tensor_access("layer0_weight", 450, timestamp=1.0)
    manager.pin_tensor("layer0_weight")

    assert manager.is_under_pressure()
    result = manager.evict_to_budget()

    # Since it was pinned, it cannot be evicted
    assert result.freed_bytes == 0
    assert "layer0_weight" not in result.evicted_tensors
    assert manager.current_allocated_bytes == 450
