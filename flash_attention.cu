#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <assert.h>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)        \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";  \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// ============================================================================
// 1. NAIVE ATTENTION KERNELS (Materializes N x N Matrix in HBM)
// ============================================================================

__global__ void naive_matmul_qk_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ S,
    const int N,
    const int d,
    const float scale
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float val = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < d; ++k) {
            val += Q[(size_t)row * d + k] * K[(size_t)col * d + k];
        }
        S[(size_t)row * N + col] = val * scale;
    }
}

__global__ void naive_softmax_kernel(
    float* __restrict__ S,
    const int N
) {
    const int row = blockIdx.x;
    if (row >= N) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // 1. Find local max per thread, then reduce across block
    float max_val = -INFINITY;
    for (int i = tid; i < N; i += num_threads) {
        max_val = fmaxf(max_val, S[(size_t)row * N + i]);
    }

    extern __shared__ float s_mem[];
    float* s_max = s_mem; // Size: blockDim.x
    s_max[tid] = max_val;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }
    const float global_max = s_max[0];
    __syncthreads();

    // 2. Compute exponentials and sum
    float sum_val = 0.0f;
    for (int i = tid; i < N; i += num_threads) {
        float val = __expf(S[(size_t)row * N + i] - global_max);
        S[(size_t)row * N + i] = val; // Store exp temporarily
        sum_val += val;
    }

    float* s_sum = s_mem; 
    s_sum[tid] = sum_val;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    const float global_sum = s_sum[0];
    __syncthreads();

    // 3. Normalize to construct Softmax distribution
    const float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 1.0f;
    for (int i = tid; i < N; i += num_threads) {
        S[(size_t)row * N + i] *= inv_sum;
    }
}

__global__ void naive_matmul_pv_kernel(
    const float* __restrict__ P,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int N,
    const int d
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < d) {
        float val = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < N; ++k) {
            val += P[(size_t)row * N + k] * V[(size_t)k * d + col];
        }
        O[(size_t)row * d + col] = val;
    }
}

// ============================================================================
// 2. FLASH ATTENTION KERNEL (Online Softmax with Register/Shared Memory Tiling)
// ============================================================================

__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int N,
    const int d,
    const float scale
) {
    assert(d == 64 && "This kernel only supports d=64");

    // Shared memory with +4 padding to maintain 16-byte alignment and bypass bank conflicts
    __shared__ float s_Q[32][64 + 4];
    __shared__ float s_K[32][64 + 4];
    __shared__ float s_V[32][64 + 4];

    const int tx = threadIdx.x; 
    const int warp_id = tx / 32;
    const int lane_id = tx % 32;
    const int row_block_idx = blockIdx.x; 

    // Each block handles 4 rows (128 threads / 32 threads-per-row)
    const int global_row = row_block_idx * (blockDim.x / 32) + warp_id;

    // Registers for running metrics
    float r_m = -INFINITY;
    float r_l = 0.0f;
    float r_O[64];
    
    #pragma unroll
    for (int c = 0; c < 64; ++c) {
        r_O[c] = 0.0f;
    }

    // Load static Q tile into shared memory cooperatively
    if (global_row < N) {
        s_Q[warp_id][lane_id] = Q[(size_t)global_row * d + lane_id];
        s_Q[warp_id][lane_id + 32] = Q[(size_t)global_row * d + lane_id + 32];
    } else {
        s_Q[warp_id][lane_id] = 0.0f;
        s_Q[warp_id][lane_id + 32] = 0.0f;
    }
    __syncthreads();

    const int num_cols_blocks = (N + 31) / 32;

    for (int col_block = 0; col_block < num_cols_blocks; ++col_block) {
        // Load K and V tiles cooperatively (using the first warp of 32 threads)
        if (tx < 32) {
            const int k_row = col_block * 32 + tx;
            if (k_row < N) {
                #pragma unroll
                for (int c = 0; c < 64; ++c) {
                    s_K[tx][c] = K[(size_t)k_row * d + c];
                    s_V[tx][c] = V[(size_t)k_row * d + c];
                }
            } else {
                #pragma unroll
                for (int c = 0; c < 64; ++c) {
                    s_K[tx][c] = 0.0f;
                    s_V[tx][c] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Calculate scores S_ij = (Q_i * K_j^T) * scale
        float S[32];
        // Outer loop unroll removed to prevent excessive instruction cache expansion
        for (int k = 0; k < 32; ++k) {
            float score = 0.0f;
            #pragma unroll
            for (int c = 0; c < 64; ++c) {
                score += s_Q[warp_id][c] * s_K[k][c];
            }
            S[k] = score * scale;

            int global_k = col_block * 32 + k;
            if (global_k >= N) S[k] = -1e9f;
        }

        // Evaluate local tile max
        float tile_m = -INFINITY;
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            tile_m = fmaxf(tile_m, S[k]);
        }

        // Evaluate local tile denominator
        float tile_l = 0.0f;
        float exp_S[32];
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            exp_S[k] = __expf(S[k] - tile_m);
            tile_l += exp_S[k];
        }

        // Compute online scaling parameters
        const float next_m = fmaxf(r_m, tile_m);
        const float scale_old = (next_m == -INFINITY) ? 0.0f : __expf(r_m - next_m);
        const float scale_tile = (next_m == -INFINITY) ? 0.0f : __expf(tile_m - next_m);

        // Update softmax denominator scaling
        r_l = r_l * scale_old + tile_l * scale_tile;

        // Scale running output accumulator
        #pragma unroll
        for (int c = 0; c < 64; ++c) {
            r_O[c] *= scale_old;
        }

        // Accumulate weighted V values
        // Outer loop unroll removed to prevent instruction footprint bloat and register spilling
        for (int k = 0; k < 32; ++k) {
            const float weight = exp_S[k] * scale_tile;
            #pragma unroll
            for (int c = 0; c < 64; ++c) {
                r_O[c] += weight * s_V[k][c];
            }
        }

        r_m = next_m;
        __syncthreads();
    }

    // Finalize division by cumulative softmax denominator
    if (global_row < N) {
        const float inv_l = (r_l > 0.0f) ? (1.0f / r_l) : 1.0f;
        #pragma unroll
        for (int c = 0; c < 64; ++c) {
            O[(size_t)global_row * d + c] = r_O[c] * inv_l;
        }
    }
}

// ============================================================================
// 3. HOST-SIDE REFERENCE IMPLEMENTATION & VALIDATION PROCESSOR
// ============================================================================

void cpu_attention_reference(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const int N,
    const int d
) {
    const float scale = 1.0f / sqrtf(static_cast<float>(d));
    std::vector<float> S(N * N);
    std::vector<float> P(N * N);

    // Q * K^T
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < d; ++k) {
                sum += Q[i * d + k] * K[j * d + k];
            }
            S[i * N + j] = sum * scale;
        }
    }

    // Row-wise Softmax
    for (int i = 0; i < N; ++i) {
        float max_val = -INFINITY;
        for (int j = 0; j < N; ++j) {
            max_val = std::max(max_val, S[i * N + j]);
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            P[i * N + j] = std::exp(S[i * N + j] - max_val);
            sum_exp += P[i * N + j];
        }
        for (int j = 0; j < N; ++j) {
            P[i * N + j] /= sum_exp;
        }
    }

    // P * V
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += P[i * N + k] * V[k * d + j];
            }
            O[i * d + j] = sum;
        }
    }
}

bool verify_correctness(const float* ref, const float* test, int size, float tolerance = 1e-3f) {
    float max_diff = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = std::abs(ref[i] - test[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    std::cout << "  Verification: Maximum Absolute Deviation = " << max_diff << " -> ";
    return max_diff < tolerance;
}

void execute_benchmark(const int N, const int d) {
    std::cout << "\n========================================\n";
    std::cout << "BENCHMARK CONFIGURATION: N = " << N << ", d = " << d << "\n";
    std::cout << "========================================\n";

    const size_t qkv_size = (size_t)N * d * sizeof(float);
    const size_t att_size = (size_t)N * N * sizeof(float);

    std::vector<float> h_Q(N * d);
    std::vector<float> h_K(N * d);
    std::vector<float> h_V(N * d);
    std::vector<float> h_O_naive(N * d, 0.0f);
    std::vector<float> h_O_flash(N * d, 0.0f);
    std::vector<float> h_O_ref(N * d, 0.0f);

    // Initialize inputs with deterministic pattern
    for (int i = 0; i < N * d; ++i) {
        h_Q[i] = static_cast<float>(i % 7 - 3) * 0.1f;
        h_K[i] = static_cast<float>(i % 5 - 2) * 0.1f;
        h_V[i] = static_cast<float>(i % 9 - 4) * 0.1f;
    }

    // Allocate GPU Space
    float *d_Q, *d_K, *d_V, *d_S, *d_O_naive, *d_O_flash;
    CHECK_CUDA(cudaMalloc(&d_Q, qkv_size));
    CHECK_CUDA(cudaMalloc(&d_K, qkv_size));
    CHECK_CUDA(cudaMalloc(&d_V, qkv_size));
    CHECK_CUDA(cudaMalloc(&d_S, att_size)); // Intermediary global score allocation for naive
    CHECK_CUDA(cudaMalloc(&d_O_naive, qkv_size));
    CHECK_CUDA(cudaMalloc(&d_O_flash, qkv_size));

    // Host-to-Device Copy
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), qkv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), qkv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), qkv_size, cudaMemcpyHostToDevice));

    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    // Timing events
    cudaEvent_t start, stop;
    float elapsed_ms;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ------------------------------------------------------------------------
    // NAIVE ATTENTION PROFILING RUN
    // ------------------------------------------------------------------------
    dim3 block_matmul(16, 16);
    dim3 grid_matmul_qk((N + 15) / 16, (N + 15) / 16);
    dim3 grid_matmul_pv((d + 15) / 16, (N + 15) / 16);
    int threads_softmax = 256;
    size_t shared_softmax_bytes = threads_softmax * sizeof(float);

    CHECK_CUDA(cudaEventRecord(start));
    
    naive_matmul_qk_kernel<<<grid_matmul_qk, block_matmul>>>(d_Q, d_K, d_S, N, d, scale);
    CHECK_CUDA(cudaGetLastError());
    
    naive_softmax_kernel<<<N, threads_softmax, shared_softmax_bytes>>>(d_S, N);
    CHECK_CUDA(cudaGetLastError());
    
    naive_matmul_pv_kernel<<<grid_matmul_pv, block_matmul>>>(d_S, d_V, d_O_naive, N, d);
    CHECK_CUDA(cudaGetLastError());
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "  Naive Attention Run-time: " << std::fixed << std::setprecision(4) << elapsed_ms << " ms\n";
    CHECK_CUDA(cudaMemcpy(h_O_naive.data(), d_O_naive, qkv_size, cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------------
    // FLASH ATTENTION PROFILING RUN
    // ------------------------------------------------------------------------
    dim3 block_flash(128); // Each block processes 4 rows with 128 threads total (32 threads per row)
    dim3 grid_flash((N + (block_flash.x / 32) - 1) / (block_flash.x / 32)); 

    CHECK_CUDA(cudaEventRecord(start));
    
    flash_attention_kernel<<<grid_flash, block_flash>>>(d_Q, d_K, d_V, d_O_flash, N, d, scale);
    CHECK_CUDA(cudaGetLastError());
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "  Flash Attention Run-time: " << std::fixed << std::setprecision(4) << elapsed_ms << " ms\n";
    CHECK_CUDA(cudaMemcpy(h_O_flash.data(), d_O_flash, qkv_size, cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------------
    // CPU VERIFICATION
    // ------------------------------------------------------------------------
    std::cout << "  Executing CPU Reference Computation...\n";
    cpu_attention_reference(h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(), N, d);

    bool naive_pass = verify_correctness(h_O_ref.data(), h_O_naive.data(), N * d);
    std::cout << (naive_pass ? "PASS" : "FAIL") << "\n";

    bool flash_pass = verify_correctness(h_O_ref.data(), h_O_flash.data(), N * d);
    std::cout << (flash_pass ? "PASS" : "FAIL") << "\n";

    // Cleanup Resources
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_S));
    CHECK_CUDA(cudaFree(d_O_naive));
    CHECK_CUDA(cudaFree(d_O_flash));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    execute_benchmark(512, 64);
    execute_benchmark(1024, 64);
    execute_benchmark(2048, 64);
    return 0;
}
