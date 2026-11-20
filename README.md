# 🚀 Distributed Memory Grid for Real-Time LLM Inference
### Zaalima Development – Project 1 | Member 2: LLM Model Loading & Memory Mapping

An enterprise-grade Python module for **zero-copy memory-mapped Hugging Face Safetensors weight loading**, **architectural configuration validation**, **multi-node tensor-parallel weight sharding**, and **cryptographic/numerical weight integrity verification**.

---

## 📂 Project Architecture

```text
Distributed Memory Grid for Real-Time LLM Inference/
├── pyproject.toml
├── requirements.txt
├── README.md
├── src/
│   └── dmg/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── exceptions.py          # Enterprise exception hierarchy (DMGError, ShapeMismatchError, etc.)
│       │   └── interfaces.py          # Cross-team protocols & data contracts (Members 1–6)
│       └── model_loader/              # Member 2 Core Deliverables
│           ├── __init__.py
│           ├── config.py              # LLMModelConfig & ConfigValidator (MHA/GQA/MQA, RoPE, TP divisibility)
│           ├── safetensors_io.py      # Zero-allocation Safetensors binary header parser & multi-shard index resolver
│           ├── mmap_loader.py         # OS virtual memory-mapped (mmap) zero-copy tensor store & 2D axis slicer
│           ├── loader.py              # Optimized LLMWeightLoader (thread-pool prefetch, LRU cache, dtype cast)
│           ├── sharding.py            # WeightDistributor for Member 1 (nodes), Member 3 (Attention), Member 4 (FFN)
│           └── validator.py           # TensorIntegrityValidator (shapes, NaN/Inf, SHA-256 & CRC32 verification)
└── tests/
    ├── conftest.py                    # Synthetic single-shard & multi-shard Safetensors checkpoint fixtures
    ├── test_config_and_safetensors.py # Week 1: Config & Safetensors header/index unit tests
    ├── test_mmap_loader.py            # Week 1: Memory-mapped zero-copy store & foundational loader tests
    ├── test_sharding_and_dist.py      # Week 2: Attention & FFN node sharding and lossless reconstruction tests
    └── test_validator_and_pipeline.py # Week 2: Integrity validation & end-to-end optimized pipeline tests
```

---

## 🔗 Team Integration Contracts (Members 1–6)

| Team Member | Integration Interface Provided by Member 2 | Purpose |
| :--- | :--- | :--- |
| **Member 1** *(Master-Worker Network)* | `WorkerShardTarget`, `NodeShardPackage`, `LLMWeightLoader.prepare_distributed_shards()` | Generates per-worker weight packages with SHA-256 manifests and `.safetensors` shard exports for gRPC distribution. |
| **Member 3** *(Attention Parallelism)* | `AttentionParallelismProvider.get_attention_shard_spec()` | Column-parallel head splitting for `q_proj`, `k_proj`, `v_proj` (supporting MHA and GQA) and row-parallel splitting for `o_proj`. |
| **Member 4** *(Feed-Forward Distribution)* | `FFNParallelismProvider.get_ffn_shard_spec()` | Column-parallel splitting for `gate_proj` and `up_proj`, and row-parallel splitting for `down_proj`. |
| **Member 5** *(Async KV-Cache)* | `LLMModelConfig.head_dim`, `num_key_value_heads`, `kv_hidden_size` | Supplies validated per-layer KV head dimensions and dtype byte widths for non-fragmented KV page allocation. |
| **Member 6** *(FastAPI Gateway)* | `LLMWeightLoader.validate_checkpoint()`, `LLMModelConfig` | Validates model readiness, context length (`max_position_embeddings`), and memory telemetry prior to serving requests. |

---

## ⚡ Quick Start & Usage

### 1. Zero-Copy Memory-Mapped Weight Loading & Validation

```python
from dmg.core.interfaces import WorkerShardTarget
from dmg.model_loader import LLMWeightLoader

with LLMWeightLoader(
    checkpoint_dir="./checkpoints/llama-model",
    cache_capacity_tensors=32,
    prefetch_workers=4,
    target_dtype="F16",
) as loader:
    # 1. Validate all tensor shapes, NaN/Inf finiteness, and SHA-256 checksums
    report = loader.validate_checkpoint()
    assert report.passed

    # 2. Asynchronous background layer prefetching
    loader.prefetch_layers([0, 1])
    layer_0 = loader.load_layer(0)

    # 3. Prepare distributed shards across 4 worker nodes (Members 1, 3, 4)
    workers = [
        WorkerShardTarget(node_id=f"worker-{r}", rank=r, world_size=4)
        for r in range(4)
    ]
    node_packages = loader.prepare_distributed_shards(workers, validate=True)
```

---

## 🧪 Running the Test Suite

```bash
pytest -v
```
