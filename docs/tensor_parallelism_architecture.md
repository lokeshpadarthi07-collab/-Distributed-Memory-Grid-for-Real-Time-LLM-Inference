# Architecture Specification: Tensor Parallelism for Distributed LLM Inference
**Module:** Member 3 – Attention Tensor Parallelism  
**System:** Distributed Memory Grid (DMG) for Real-Time LLM Inference

---

## 1. Study of PyTorch Distributed Architecture

PyTorch distributed (`torch.distributed`) provides primitive constructs for multi-GPU and multi-node execution:

### 1.1 ProcessGroup & Backends
- **NCCL (NVIDIA Collective Communications Library):** Standard high-performance backend for GPU-to-GPU clusters over NVLink and InfiniBand.
- **Gloo:** Multi-platform backend supporting CPU and GPU tensors, ideal for CPU inference nodes, fallback environments, and testing.
- **MPI:** Message Passing Interface for HPC clusters.

### 1.2 Collective Communication Primitives
Tensor Parallelism (Megatron-LM style) relies on specific collective operations:
1. **All-Reduce ($\Sigma$):** Every node provides a local tensor; all nodes receive the element-wise reduction (sum or average) across all ranks. Used at the end of Row-Parallel Linear layers and Attention output projections.
2. **All-Gather ($[\cdot]$):** Concatenates tensors from all ranks along a specified dimension. Every node receives the full aggregated tensor.
3. **Reduce-Scatter:** Combines reduction and scattering: element-wise reduces input across ranks, then scatters equal chunks to each rank.
4. **Broadcast / Scatter:** One master rank sends data to all worker ranks.

### 1.3 Megatron-LM Tensor Parallelism Paradigm for Transformer Models
In standard multi-head self-attention:
- **Projection Layer ($W_q, W_k, W_v$):** Partitioned using **Column-Parallelism** (splitting along output hidden dimension / number of heads $H$).
  $$\left[\begin{array}{c} Q_i \\ K_i \\ V_i \end{array}\right] = X \cdot \left[\begin{array}{ccc} W_{q,i} & W_{k,i} & W_{v,i} \end{array}\right]$$
  No communication is needed before or during the localized attention computation!
- **Attention Computation:** Each rank $i$ independently executes scaled dot-product attention on its local heads $H / P$:
  $$O_i = \text{Softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$
- **Output Projection Layer ($W_o$):** Partitioned using **Row-Parallelism** (splitting along input hidden dimension):
  $$O = \sum_{i=1}^P O_i W_{o,i} = \text{All-Reduce}_{\text{sum}}\left(O_i W_{o,i}\right)$$
  A single `All-Reduce` operation at the end of the attention block synchronizes the output across all $P$ ranks.

---

## 2. Distributed Memory Grid (DMG) Module Architecture

Member 3 fits into the overall DMG pipeline between Member 2 (Model Weight Loader) and Member 1 (Master-Worker Network):

```
+-----------------------------------------------------------+
|          Member 2: Weight Loader & Sharding Store         |
+-----------------------------------------------------------+
                             |
         Attention Weights (W_q, W_k, W_v, W_o)
                             v
+-----------------------------------------------------------+
|      Member 3: Tensor Parallelism (Attention Heads)       |
|                                                           |
|  1. AttentionHeadSharder:                                 |
|     - MHA / GQA / MQA head-to-rank slicing                |
|  2. ColumnParallelLinear & RowParallelLinear:             |
|     - Weight splitting along column and row dims          |
|  3. ParallelSelfAttention:                                |
|     - Local Q, K, V projection & Scaled Dot-Product       |
|  4. AttentionOutputSynchronizer:                          |
|     - All-Reduce cross-rank synchronization & residual    |
+-----------------------------------------------------------+
                             |
                   Synchronized Activations
                             v
+-----------------------------------------------------------+
|       Member 4: Feed-Forward Network (FFN) Distribution   |
+-----------------------------------------------------------+
```

## 3. Supported Attention Architectures
- **Multi-Head Attention (MHA):** Equal number of Query, Key, and Value heads ($H_q = H_k = H_v$). Heads divided evenly across ranks ($H / P$).
- **Grouped-Query Attention (GQA):** LLaMA-2 / LLaMA-3 / Mistral style. Query heads partitioned into groups sharing a smaller number of KV heads.
- **Multi-Query Attention (MQA):** Multiple Query heads share a single Key and Value head across all ranks.
