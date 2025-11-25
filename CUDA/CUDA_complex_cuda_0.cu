// CUDA_complex_cuda_0.cu
// 使用实际路径替换
// nvcc -O3 -arch=sm_80 -lineinfo -I/usr/local/cuda-13.0/targets/x86_64-linux/include/nvtx3/nvToolsExt.h -L/usr/local/cuda/lib64 -DUSE_NVTX -o CUDA_complex_cuda_0 CUDA_complex_cuda_0.cu -lnvToolsExt

// nvcc -O3 -arch=sm_80 -lineinfo \
// -I/usr/local/cuda-13.0/targets/x86_64-linux/include/nvtx3/nvToolsExt.h \
// -L/usr/local/cuda/lib64 \
// -DUSE_NVTX -o CUDA_complex_cuda_0 CUDA_complex_cuda_0.cu -lnvToolsExt


#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#ifdef USE_NVTX
#include <nvToolsExt.h>
#endif

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Kernel: Vector Add
__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Kernel: Matrix Add with bias
__global__ void matAddBias(float* C, const float* bias, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        C[row*N+col] += bias[row];
    }
}

// Kernel: Matrix Multiply (tiled)
template <int TILE>
__global__ void matMulTiled(const float* A, const float* B, float* C, int N) {
    __shared__ float sA[TILE][TILE], sB[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float sum = 0.0f;
    for (int t = 0; t < N; t += TILE) {
        sA[ty][tx] = (row < N && t+tx < N) ? A[row*N + t + tx] : 0.0f;
        sB[ty][tx] = (t+ty < N && col < N) ? B[(t+ty)*N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) sum += sA[ty][k] * sB[k][tx];
        __syncthreads();
    }
    if (row < N && col < N) C[row*N+col] = sum;
}

// Kernel: block reduction
__global__ void blockReduceSum(const float* in, float* partial, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x, gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float val = 0.0f;
    if (gid < n) val = in[gid];
    if (gid + blockDim.x < n) val += in[gid + blockDim.x];
    sdata[tid] = val;
    __syncthreads();
    for (unsigned int s = blockDim.x/2; s; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// 验证函数
bool nearly_equal(float a, float b, float eps=1e-3) {
    return std::fabs(a-b) < eps;
}

int main() {
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Using GPU: %s\n", prop.name);

    // --- VecAdd Test ---
    const int Nvec = 1<<20;
    float *h_a, *h_b, *h_c, *h_c_cpu;
    CUDA_CHECK(cudaHostAlloc(&h_a, Nvec*sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_b, Nvec*sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_c, Nvec*sizeof(float), cudaHostAllocDefault));
    h_c_cpu = (float*)malloc(Nvec*sizeof(float));
    for (int i = 0; i < Nvec; ++i) {
        h_a[i] = 1.0f; h_b[i] = 2.0f; h_c[i] = 0.0f; h_c_cpu[i] = h_a[i] + h_b[i];
    }
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, Nvec*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, Nvec*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, Nvec*sizeof(float)));

#ifdef USE_NVTX
    nvtxRangePushA("VecAdd");
#endif
    CUDA_CHECK(cudaMemcpy(d_a, h_a, Nvec*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, Nvec*sizeof(float), cudaMemcpyHostToDevice));
    vecAdd<<<(Nvec+255)/256, 256>>>(d_a, d_b, d_c, Nvec);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_c, d_c, Nvec*sizeof(float), cudaMemcpyDeviceToHost));
#ifdef USE_NVTX
    nvtxRangePop();
#endif
    // correctness check
    bool correct = true;
    for (int i = 0; i < Nvec; ++i) if (!nearly_equal(h_c[i], h_c_cpu[i])) { correct = false; break; }
    printf("VecAdd: correct=%s\n", correct ? "YES" : "NO");

    // --- Matrix Multiply Test ---
    const int N = 512;
    float *h_MA = (float*)malloc(N*N*sizeof(float));
    float *h_MB = (float*)malloc(N*N*sizeof(float));
    float *h_MC = (float*)malloc(N*N*sizeof(float));
    float *h_MC_cpu = (float*)malloc(N*N*sizeof(float));
    for (int i = 0; i < N*N; ++i) { h_MA[i] = 1.0f; h_MB[i] = 2.0f; h_MC[i] = 0.0f; }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            h_MC_cpu[i*N+j] = 0.0f;
    // CPU reference
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                h_MC_cpu[i*N+j] += h_MA[i*N+k] * h_MB[k*N+j];

    float *d_MA, *d_MB, *d_MC;
    CUDA_CHECK(cudaMalloc(&d_MA, N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_MB, N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_MC, N*N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_MA, h_MA, N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_MB, h_MB, N*N*sizeof(float), cudaMemcpyHostToDevice));

#ifdef USE_NVTX
    nvtxRangePushA("MatMulTiled");
#endif
    dim3 threads(16,16), blocks((N+15)/16,(N+15)/16);
    matMulTiled<16><<<blocks, threads>>>(d_MA, d_MB, d_MC, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_MC, d_MC, N*N*sizeof(float), cudaMemcpyDeviceToHost));
#ifdef USE_NVTX
    nvtxRangePop();
#endif
    // correctness check
    correct = true;
    for (int i = 0; i < N*N; ++i) if (!nearly_equal(h_MC[i], h_MC_cpu[i])) { correct = false; break; }
    printf("MatMulTiled: correct=%s\n", correct ? "YES" : "NO");

    // --- Reduction Test ---
    const int Nred = 1<<20;
    float *h_red = (float*)malloc(Nred*sizeof(float));
    float *h_partial = (float*)malloc(4096*sizeof(float));
    for (int i = 0; i < Nred; ++i) h_red[i] = 1.0f;
    float *d_red, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_red, Nred*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, 4096*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_red, h_red, Nred*sizeof(float), cudaMemcpyHostToDevice));

#ifdef USE_NVTX
    nvtxRangePushA("Reduction");
#endif
    int threadsR = 256, blocksR = (Nred+threadsR*2-1)/(threadsR*2);
    blockReduceSum<<<blocksR, threadsR, threadsR*sizeof(float)>>>(d_red, d_partial, Nred);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, blocksR*sizeof(float), cudaMemcpyDeviceToHost));
#ifdef USE_NVTX
    nvtxRangePop();
#endif

    float total_gpu = 0.0f;
    for (int i = 0; i < blocksR; ++i) total_gpu += h_partial[i];
    printf("Reduction: sum=%f (expected %f)\n", total_gpu, (float)Nred);

    // --- Matrix Add with bias Test ---
    float *h_bias = (float*)malloc(N*sizeof(float));
    for (int i = 0; i < N; ++i) h_bias[i] = (float)i;
    float *d_bias;
    CUDA_CHECK(cudaMalloc(&d_bias, N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, N*sizeof(float), cudaMemcpyHostToDevice));
    matAddBias<<<blocks, threads>>>(d_MC, d_bias, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_MC, d_MC, N*N*sizeof(float), cudaMemcpyDeviceToHost));
    printf("MatAddBias: MC[0]=%f (bias=%f)\n", h_MC[0], h_bias[0]);

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_MA)); CUDA_CHECK(cudaFree(d_MB)); CUDA_CHECK(cudaFree(d_MC));
    CUDA_CHECK(cudaFree(d_red)); CUDA_CHECK(cudaFree(d_partial)); CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFreeHost(h_a)); CUDA_CHECK(cudaFreeHost(h_b)); CUDA_CHECK(cudaFreeHost(h_c));
    free(h_MA); free(h_MB); free(h_MC); free(h_MC_cpu); free(h_red); free(h_partial); free(h_bias);
    printf("Done.\n");
    return 0;
}
