"""
Complete demonstration script for Member 3: Tensor Parallelism (Attention Heads).
Simulates a 2-node distributed cluster running parallel multi-head self-attention
with KV-caching, All-Reduce synchronization, and latency profiling.
"""

from __future__ import annotations

import pprint

from dmg.tensor_parallel.attention import (
    ColumnParallelLinear,
    ParallelSelfAttention,
    RowParallelLinear,
)
from dmg.tensor_parallel.communicator import MockCommunicator, MockCommunicatorGroup
from dmg.tensor_parallel.config import (
    AttentionType,
    CommBackend,
    ParallelAttentionConfig,
    TensorParallelConfig,
)
from dmg.tensor_parallel.engine import DistributedInferenceEngine
from dmg.tensor_parallel.kv_cache import DistributedKVCacheManager
from dmg.tensor_parallel.sharding import AttentionHeadSharder
from dmg.tensor_parallel.synchronizer import AttentionOutputSynchronizer


def main():
    print("=" * 60)
    print("DMG Member 3: Distributed Attention Tensor Parallelism Demo")
    print("=" * 60)

    # Cluster configuration: 2 worker ranks, 4 heads, head_dim 2 (hidden_size 8)
    world_size = 2
    hidden_size = 8
    num_heads = 4
    head_dim = 2

    group = MockCommunicatorGroup(world_size=world_size)
    engines = []

    # Initialize each rank node
    for rank in range(world_size):
        tp_cfg = TensorParallelConfig(
            world_size=world_size,
            rank=rank,
            backend=CommBackend.MOCK,
            enable_profiling=True,
        )
        att_cfg = ParallelAttentionConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            attention_type=AttentionType.MHA,
            tp_config=tp_cfg,
        )

        comm = MockCommunicator(tp_cfg, shared_group=group)
        sharder = AttentionHeadSharder(att_cfg)
        part = sharder.compute_head_partition(rank, world_size)

        print(
            f"[Node Rank {rank}] Query Heads: {part.query_heads_range}, "
            f"KV Heads: {part.kv_heads_range}, Local Dim: {att_cfg.local_hidden_size}"
        )

        # Mock local weights
        w_q = [[0.5] * hidden_size for _ in range(att_cfg.local_hidden_size)]
        w_k = [[0.5] * hidden_size for _ in range(att_cfg.local_hidden_size)]
        w_v = [[0.5] * hidden_size for _ in range(att_cfg.local_hidden_size)]
        w_o = [[0.25] * att_cfg.local_hidden_size for _ in range(hidden_size)]

        q_proj = ColumnParallelLinear(w_q)
        k_proj = ColumnParallelLinear(w_k)
        v_proj = ColumnParallelLinear(w_v)
        o_proj = RowParallelLinear(w_o)

        attention = ParallelSelfAttention(att_cfg, q_proj, k_proj, v_proj, o_proj)
        synchronizer = AttentionOutputSynchronizer(comm, bias=[0.05] * hidden_size)

        engine = DistributedInferenceEngine(
            config=att_cfg,
            attention_layers=[attention],
            synchronizers=[synchronizer],
        )
        engines.append(engine)

    # Common input prompt tokens (2 tokens x 8 dimensions)
    prompt = [
        [1.0, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.9, 0.4, 0.3, 0.2, 0.1, 0.0],
    ]

    print("\n--- Running Distributed Forward Pass ---")
    step_id = "demo_step_1"

    # Step 1: Local computation & submission
    for r in range(world_size):
        local_out = engines[r].attention_layers[0].forward_local(prompt)
        engines[r].synchronizers[0].communicator.submit_tensor(
            f"step_{step_id}_layer_0", local_out
        )

    # Step 2: Cross-rank output synchronization
    outputs = []
    for r in range(world_size):
        out = engines[r].forward_layer(
            layer_idx=0,
            hidden_states=prompt,
            residual=prompt,
            step_id=step_id,
        )
        outputs.append(out)

    print("Rank 0 Output Shape:", len(outputs[0]), "tokens x", len(outputs[0][0]), "dims")
    print("Rank 0 and Rank 1 Outputs Identical?", outputs[0] == outputs[1])

    print("\n--- Profiling Summary ---")
    pprint.pprint(engines[0].profiler.get_summary())
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
