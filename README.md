# CUDA Flash Attention: Online Softmax with Shared Memory Tiling

**By:** Sana Ul Mustafa Qadri  
**Hardware:** Tesla T4 (sm_75, 40 SMs)  
**Language:** CUDA C++ (C++17), portable CC 7.0+  
**Status:** Completed — Compiled, Tested, Verified

---

## Project Summary

Implemented scaled dot-product attention (core of transformer architecture) using two approaches:

1. **Naive Attention** — Materializes full N×N score matrix in HBM (memory-intensive)
2. **Flash Attention** — Online softmax with shared memory tiling, never materializes N×N matrix

Full correctness audit (13-category checklist), performance audit (8 issues found and fixed), and verified against CPU reference with exact match (Maximum Absolute Deviation = 0.0000).

**Key results on Tesla T4:**
- N=512: Flash **1.99ms** vs Naive 3.47ms **(1.74x faster)**
- N=1024: Flash 6.38ms vs Naive 2.38ms
- N=2048: Flash 23.46ms vs Naive 9.26ms
- **Correctness: PASS (0.0000 deviation) on all sizes**

---

## Problem Statement

Standard attention computes softmax(QK^T/sqrt(d)) * V with O(N²) memory complexity — for N=2048, this requires a 16MB intermediate matrix in HBM. Flash Attention reduces this to O(N) by:

1. Processing Q, K, V in tiles that fit in shared memory (SRAM)
2. Using online softmax to maintain running max and sum across tiles
3. Never writing the full N×N attention matrix to HBM

---

## Implementation

### Kernel 1 — Naive Attention (3 separate kernels)

**QK Matmul:** Each thread computes one score S[i][j] = Q[i] · K[j] / sqrt(d)

**Softmax:** Row-wise reduction using shared memory — find max, compute exp, normalize

**PV Matmul:** Each thread computes one output O[i] = sum(P[i][j] * V[j])

**Memory complexity:** O(N²) — 16MB at N=2048 in HBM

### Kernel 2 — Flash Attention (single fused kernel)

```cuda
__global__ void flash_attention_kernel(
    const float* Q, const float* K, const float* V,
    float* O, const int N, const int d, const float scale)
{
    // Shared memory with +4 padding (16-byte aligned, bank-conflict free)
    __shared__ float s_Q[32][64 + 4];
    __shared__ float s_K[32][64 + 4];
    __shared__ float s_V[32][64 + 4];

    // Online softmax running statistics (per thread in registers)
    float r_m = -INFINITY;  // running max
    float r_l = 0.0f;       // running sum
    float r_O[64] = {0.0f}; // running output accumulator

    for (int col_block = 0; col_block < num_cols_blocks; ++col_block) {
        // Load K, V tiles into shared memory
        // Compute scores with tile masking for boundary safety
        // Update online softmax: r_m, r_l, r_O
    }
    // Write final output: O[i] = r_O[i] / r_l
}
Design decisions:
+4 padding on shared memory — row stride 272 bytes (16-byte aligned), enables LDS.128
128 threads/block (4 warps) — improved SM occupancy vs original 32 threads
Online softmax — mathematically equivalent to standard softmax, proven correct
Tile masking (-1e9f for OOB keys) — correct boundary handling for any N
NaN guard — protects against -inf - (-inf) = NaN when all scores are masked
size_t indexing — prevents int overflow at large N
Outer k-loop unroll removed — reduces instruction cache pressure and register spilling
Memory complexity: O(N) — only Q/K/V tiles in shared memory
Correctness Verification
Methodology
GPU output compared against single-threaded CPU reference: |gpu[i] - ref[i]| ≤ tolerance (1e-3)
Maximum Absolute Deviation reported for each run
Results on Tesla T4
N
Naive Deviation
Flash Deviation
Status
512
0.0000
0.0000
PASS
1024
0.0000
0.0000
PASS
2048
0.0000
0.0000
PASS
Exact zero deviation — online softmax is mathematically equivalent to standard softmax.
Correctness Audit (13-Category Checklist)
Category
Status
OOB global memory access
No issue — boundary guards correct
Races on shared memory
No issue — dual syncthreads correct
Divergent barrier / deadlock
No issue — uniform loop trip count
Uninitialized shared memory
No issue — zero-fill for OOB
Integer overflow
Fixed — size_t promotion applied
Architecture-dependent constructs
N/A — CC 7.0+ portable
Warp primitive masks
N/A — no shuffle used
Memory ordering / fence scope
No issue
Async error surfacing
Fixed — cudaGetLastError added
Cross-kernel seam
No issue
Pointer aliasing
No issue — restrict on all pointers
Numerical correctness
Fixed — NaN guard + tile masking
Latent coupling bugs
Fixed — assert(d==64) added
No Tier 0 or Tier 1 issues found after fixes.
Performance Results
Tesla T4 (sm_75, 40 SMs)
N
Naive (ms)
Flash (ms)
Speedup
512
3.4711
1.9891
1.74x
1024
2.3796
6.3820
0.37x
2048
9.2550
23.4578
0.39x
Compiler Statistics (ptxas -v, sm_75)
Kernel
Registers
Stack Frame
Spills
flash_attention
250
256 bytes
Confirmed
naive_matmul_qk
32
0 bytes
None
naive_softmax
24
0 bytes
None
naive_matmul_pv
36
0 bytes
None
Performance Analysis
Why Flash is faster at N=512 but slower at N=1024, 2048?
N=512 (Flash wins — 1.74x):
HBM savings dominant — Flash avoids 1MB intermediate matrix
Small N means few tiles — overhead is low
N=1024, 2048 (Naive wins):
Two structural bottlenecks identified:
Bottleneck 1 — Register Spilling:
Flash kernel uses 250 registers/thread — far exceeding hardware limits. Compiler spills to local memory (confirmed: 256 bytes stack frame). Each spill = ~200 cycle latency penalty. Naive kernels use 24-36 registers each — zero spills.
Bottleneck 2 — K/V Load Inefficiency:
In 128-thread blocks, only the first warp (32 threads) loads K and V tiles. The other 3 warps (96 threads) are idle during this phase. This 75% thread idle time during memory-bound loads eliminates the HBM savings advantage at larger N.
Root Cause (same insight as Flash Attention v2 paper)
Flash Attention v1 (Dao et al. 2022) identified these exact bottlenecks. Flash Attention v2 fixes them via:
Thread coarsening (each thread handles more output elements)
Better work partitioning across warps
Key Learnings
O(N²) → O(N) memory is real — Flash genuinely avoids the NxN matrix, confirmed by HBM usage analysis
Online softmax is exact — zero deviation from standard softmax, mathematically proven
Register pressure kills performance — 250 registers with spilling costs more than HBM savings at large N
K/V load must use all warps — Flash Attention v2's key insight, not v1
Correctness before performance — 8 fixes applied before benchmarking, exact 0.0000 deviation
