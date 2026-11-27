# 🚀 Distributed Memory Grid for Real-Time LLM Inference
### Zaalima Development – Project 1 | Member 2: LLM Model Loading & Memory Mapping

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Safetensors](https://img.shields.io/badge/Format-HuggingFace%20Safetensors-orange.svg)](https://github.com/huggingface/safetensors)
[![Zero-Copy](https://img.shields.io/badge/Memory-Zero--Copy%20mmap-green.svg)](https://docs.python.org/3/library/mmap.html)
[![Tensor-Parallelism](https://img.shields.io/badge/Parallelism-MHA%20%7C%20GQA%20%7C%20FFN-purple.svg)]()
[![Tests](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen.svg)]()

Production-grade Python module delivering **zero-copy memory-mapped model weight loading**, **Safetensors integration**, **tensor parallelism sharding (Attention & Feed-Forward)**, **asynchronous streaming & layer prefetching**, **INT8/INT4 weight quantization**, **OS page-fault warming**, **dynamic layer eviction**, and **numerical/cryptographic weight integrity validation**.

---

## 📋 Comprehensive Week 1 & Week 2 Deliverables Summary

### 👨‍💻 Week 1 Deliverables
1. **Safetensors Integration (`safetensors_io.py`)**
   - Zero-allocation binary header parser (`parse_safetensors_header`) reading 8-byte little-endian header lengths and JSON tensor descriptors (`dtype`, `shape`, `data_offsets`) without loading weight buffers into heap memory.
   - Multi-shard checkpoint resolver (`SafetensorsIndex`) seamlessly indexing single-file (`model.safetensors`) and multi-shard (`model.safetensors.index.json`) checkpoints.
   - Spec-compliant binary `.safetensors` serializer (`save_safetensors_file`).

2. **Model Configuration & Architectural Validation (`config.py`)**
   - `LLMModelConfig` supporting LLaMA, Mistral, Qwen, and custom Transformer models with Hugging Face `config.json` compatibility.
   - `ConfigValidator` verifying architectural invariants (MHA, GQA/MQA ratios, RoPE `head_dim` evenness, parameter counts, memory bytes, and tensor-parallel `world_size` divisibility).

3. **Memory-Mapped Weight Store (`mmap_loader.py`)**
   - `MemoryMappedWeightStore` mapping `.safetensors` shards directly into process virtual address space (`mmap.ACCESS_READ`) with `MADV_WILLNEED` page hints.
   - Zero-copy NumPy and `memoryview` buffers with 2D axis slicing (`get_tensor_slice`) for distributed tensor partitioning.

4. **Foundational Weight Loading Mechanism (`loader.py`)**
   - Structured layer-by-layer and component-level parameter extraction (`load_embeddings`, `load_layer`, `iter_layers`) implementing `WeightLoaderProtocol`.

---

### 👨‍💻 Week 2 Deliverables
1. **Optimized Weight Loading & Prefetch Engine (`loader.py`)**
   - Thread-safe bounded LRU weight cache (`cache_capacity_tensors`) with hit/miss telemetry.
   - Background `ThreadPoolExecutor` layer prefetching (`prefetch_layers`, `iter_layers`) overlapping page-fault warming with inference compute and network transfer.
   - On-the-fly dtype casting (`F32`, `F16`, `BF16`).

2. **Multi-Node Tensor Sharding (`sharding.py`)**
   - `WeightDistributor` and `NodeShardPackage` partitioning weights across worker nodes (`0..world_size-1`).
   - Column-parallel head splitting (`q_proj`, `k_proj`, `v_proj`) and row-parallel splitting (`o_proj`) for Member 3 (Attention Parallelism).
   - Column-parallel splitting (`gate_proj`, `up_proj`) and row-parallel splitting (`down_proj`) for Member 4 (Feed-Forward Distribution).
   - Per-worker `.safetensors` shard export and lossless tensor reconstruction (`reconstruct_sharded_tensor`).

3. **Tensor Integrity & Numerical Sanity Validation (`validator.py`)**
   - `TensorIntegrityValidator` validating parameter completeness, exact shapes/dtypes, absence of `NaN`/`Inf` or uninitialized zero weights.
   - Cryptographic SHA-256 and fast CRC32 checksum verification across distributed worker node packages.

4. **Advanced Performance & Integration Extensions**
   - **Cluster Memory Profiling (`diagnostics.py`)**: Computes per-rank VRAM/RAM footprints, savings ratios, and bridges NumPy to PyTorch tensors (`to_framework_tensor`).
   - **Async Tensor Streaming (`async_streamer.py`)**: Non-blocking `AsyncTensorStreamer` supporting chunked tensor streaming and lookahead pipeline prefetching.
   - **Weight Quantization (`quantization.py`)**: `WeightQuantizer` with symmetric per-tensor/per-channel INT8 and 4x memory-saving blockwise INT4 quantization.
   - **Page Cache Warming (`page_warmer.py`)**: `MemoryMappedPageWarmer` pre-faulting OS virtual pages across 4KB frames to eliminate cold-start inference latency.
   - **Dynamic Layer Eviction (`eviction.py`)**: `DynamicLayerEvictionManager` monitoring virtual memory pressure and enforcing high/low watermark budgets with tensor pinning.

---

## 📂 Project Structure

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
│       │   └── interfaces.py          # Cross-team protocols & contracts (Members 1–6)
│       └── model_loader/              # Member 2 Core Deliverables
│           ├── __init__.py
│           ├── config.py              # LLMModelConfig & ConfigValidator (MHA/GQA/MQA, RoPE, divisibility)
│           ├── safetensors_io.py      # Binary header parser & multi-shard index resolver
│           ├── mmap_loader.py         # MemoryMappedWeightStore (zero-copy mmap & 2D slicing)
│           ├── loader.py              # Optimized LLMWeightLoader (LRU cache, prefetch, dtype cast)
│           ├── sharding.py            # WeightDistributor for Members 1, 3, and 4
│           ├── validator.py           # TensorIntegrityValidator (shapes, NaN/Inf, SHA-256 & CRC32)
│           ├── diagnostics.py         # ClusterMemoryProfiler & BF16 bit converters
│           ├── async_streamer.py      # AsyncTensorStreamer & chunked streaming
│           ├── quantization.py        # INT8 per-channel & INT4 blockwise quantization
│           ├── page_warmer.py         # MemoryMappedPageWarmer (OS page pre-faulting)
│           └── eviction.py            # DynamicLayerEvictionManager (memory budget reclamation)
└── tests/
    ├── conftest.py                    # Synthetic single-shard & multi-shard test fixtures
    ├── test_config_and_safetensors.py # Week 1: Config & Safetensors header/index tests
    ├── test_mmap_loader.py            # Week 1: Memory-mapped zero-copy store tests
    ├── test_sharding_and_dist.py      # Week 2: Attention/FFN node sharding & reconstruction tests
    ├── test_validator_and_pipeline.py # Week 2: Integrity validator & end-to-end pipeline tests
    ├── test_diagnostics.py            # Diagnostics & BF16/FP32 bit-conversion tests
    ├── test_async_streamer.py         # Asynchronous streaming & prefetch tests
    ├── test_quantization.py           # INT8/INT4 quantization SNR & compression tests
    ├── test_page_warmer.py            # OS page-fault warmup tests
    └── test_eviction.py               # Memory high-watermark eviction tests
```

---

## 🔗 Cross-Team Integration Contracts (Members 1–6)

| Team Module | Integration Contract / Protocol | Description |
| :--- | :--- | :--- |
| **Member 1** *(Master-Worker Network)* | `WorkerShardTarget`, `NodeShardPackage`, `prepare_distributed_shards()` | Transmits per-worker weight slices with SHA-256 manifests and `.safetensors` exports over gRPC. |
| **Member 3** *(Attention Parallelism)* | `AttentionParallelismProvider.get_attention_shard_spec()` | Column-parallel head splitting (`q_proj`, `k_proj`, `v_proj`) and row-parallel splitting (`o_proj`). |
| **Member 4** *(Feed-Forward Distribution)* | `FFNParallelismProvider.get_ffn_shard_spec()` | Column-parallel splitting (`gate_proj`, `up_proj`) and row-parallel splitting (`down_proj`). |
| **Member 5** *(Async KV-Cache)* | `LLMModelConfig.head_dim`, `num_key_value_heads`, `kv_hidden_size` | Provides validated KV head dimensions and byte widths for zero-fragmentation cache allocation. |
| **Member 6** *(FastAPI Gateway)* | `LLMWeightLoader.validate_checkpoint()`, `LLMModelConfig` | Exposes model health check endpoints, context length limits (`max_position_embeddings`), and memory telemetry. |

---

## 🧪 Verification & Running Tests

```bash
pytest -v
```
All unit and integration test suites pass with 100% test coverage across both Week 1 and Week 2 functionality.
