# Distributed Memory Grid (DMG) for Real-Time LLM Inference

Enterprise-grade distributed memory grid and inference runtime for high-throughput Large Language Model (LLM) serving.

---

## 👨💻 MEMBER 3 – Tensor Parallelism (Attention Heads)

This repository contains the complete implementation of **Member 3 (Tensor Parallelism - Attention Heads)**, providing column/row parallel linear projections, multi-head attention partitioning (MHA, GQA, MQA), distributed scaled dot-product attention calculation, and cross-rank All-Reduce synchronization.

### 📅 6-Day Development Plan & Task Breakdown

| Day | Week | Milestone / Task | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Day 1** | Week 1 - Day 1 | **PyTorch Distributed Architecture & System Design** | Architecture document (`docs/tensor_parallelism_architecture.md`), `TensorParallelConfig`, `ParallelAttentionConfig`, custom exceptions, protocol definitions. |
| **Day 2** | Week 1 - Day 2 | **Tensor Splitting & Merging Engine** | `TensorSplitter` (Column-Parallel, Row-Parallel, Attention Head Splitting), `TensorMerger` for exact tensor reconstruction. |
| **Day 3** | Week 1 - Day 3 | **Distributed Communication & Collective Primitives** | `TorchDistributedCommunicator` (`torch.distributed`), `MockCommunicator` with thread-safe simulation, All-Reduce (SUM), All-Gather, Broadcast. |
| **Day 4** | Week 2 - Day 4 | **Attention Head Sharding Across Network Ranks** | `AttentionHeadSharder`, Head partitioning for MHA, GQA, and MQA, cross-module `AttentionParallelismProvider` contract. |
| **Day 5** | Week 2 - Day 5 | **Distributed Attention Computation Layer** | `ColumnParallelLinear`, `RowParallelLinear`, `ParallelSelfAttention` with local head scaled dot-product & causal masking. |
| **Day 6** | Week 2 - Day 6 | **Output Synchronization & End-to-End Test Suite** | `AttentionOutputSynchronizer`, `TensorParallelAttentionBlock`, end-to-end integration test suite with multi-rank parity verification. |
| **Day 7** | Week 3 - Milestone | **KV-Cache Manager, Latency Profiler & Inference Engine** | `DistributedKVCacheManager`, `TensorParallelProfiler`, `DistributedInferenceEngine`, and runnable cluster demo (`examples/tensor_parallel_demo.py`). |
| **Day 8** | Week 3 - Milestone | **Parallel RoPE & Grouped-Query Head Expansion** | `RotaryEmbedding` (per-head position rotation without cross-rank communication), `GQAHeadExpander` (repeating KV heads to match Q groups), integrated attention pipeline. |
| **Day 9** | Week 3 - Milestone | **Memory-Efficient Tiled Attention Kernel** | `OnlineSoftmaxAccumulator`, `ChunkedAttentionKernel` (FlashAttention-style tiled scaled dot-product), sliding-window attention (SWA) support. |
| **Day 10** | Week 3 - Milestone | **Paged Attention & Virtual Memory KV-Cache Block Manager** | `PhysicalBlock`, `BlockTable`, `PagedKVCacheManager` (vLLM-style virtual memory block allocation and zero-fragmentation paging per rank). |

---

## 🏗️ Module Architecture

```
src/dmg/tensor_parallel/
├── __init__.py           # Module exports and package initialization
├── config.py             # TP and Attention configuration dataclasses
├── exceptions.py         # Specialized DMG exception hierarchy
├── interfaces.py         # Communicator and Attention protocols
├── splitting.py          # Column, Row, and Head splitting/merging
├── communicator.py       # Distributed collectives (All-Reduce, All-Gather, Broadcast)
├── sharding.py           # Multi-node MHA / GQA / MQA Head Sharder
├── attention.py          # Local Parallel Self-Attention layers
└── synchronizer.py       # Cross-rank output synchronization & E2E Block
```

---

## 🧪 Running the Unit & Integration Tests

```bash
# Run with Python standard unittest discovery
export PYTHONPATH="src"
python -m unittest discover -s tests/test_tensor_parallel

# Or run with pytest
pytest tests/test_tensor_parallel -v
```
