// complex_cuda.cu
// 复杂示例：多核函数、共享内存矩阵乘法、并行归约、原子示例、streams 与异步拷贝、NVTX 区段
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#ifdef USE_NVTX
#include <nvToolsExt.h>
#endif

// Error check macro
#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while(0)

// Kernel A: simple vector add
__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Kernel B: tiled matrix multiply (shared memory) C = A * B
// square matrices of size N x N (N divisible by TILE)
template <int TILE>
__global__ void matMulTiled(const float* A, const float* B, float* C, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    for (int t = 0; t < N; t += TILE) {
        int aidx = row * N + t + tx;
        int bidx = (t + ty) * N + col;
        sA[ty][tx] = (row < N && (t + tx) < N) ? A[aidx] : 0.0f;
        sB[ty][tx] = ( (t + ty) < N && col < N) ? B[bidx] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; ++k) sum += sA[ty][k] * sB[k][tx];
        __syncthreads();
    }

    if (row < N && col < N) C[row * N + col] = sum;
}

// Kernel C: reduction (sum) per-block -> partial sums
__global__ void blockReduceSum(const float* in, float* partial, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float val = 0.0f;
    if (gid < n) val = in[gid];
    if (gid + blockDim.x < n) val += in[gid + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    // reduce in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// Kernel D: random memory updates / atomic ops to generate irregular memory pattern
__global__ void randomAtomicKernel(int* arr, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned int state = idx * 747796405u + 1;
    for (int it = 0; it < iterations; ++it) {
        // xorshift32
        state ^= (state << 13);
        state ^= (state >> 17);
        state ^= (state << 5);
        int pos = state % n;
        atomicAdd(&arr[pos], 1);
    }
}

int main(int argc, char** argv) {
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Using GPU: %s\n", prop.name);

    const int Nvec = 1<<20; // 1M floats
    const int bytes_vec = Nvec * sizeof(float);

    // Allocate host pinned memory for async transfers
    float *h_a, *h_b, *h_c;
    CUDA_CHECK(cudaHostAlloc((void**)&h_a, bytes_vec, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_b, bytes_vec, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_c, bytes_vec, cudaHostAllocDefault));

    for (int i = 0; i < Nvec; ++i) { h_a[i] = 1.0f; h_b[i] = 2.0f; h_c[i] = 0.0f; }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes_vec));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes_vec));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes_vec));

    // Create streams
    cudaStream_t s1, s2;
    CUDA_CHECK(cudaStreamCreate(&s1));
    CUDA_CHECK(cudaStreamCreate(&s2));

#ifdef USE_NVTX
    nvtxRangePushA("VecAdd-Region");
#endif

    // Asynchronous copy in stream s1 and vecAdd on s1
    CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, bytes_vec, cudaMemcpyHostToDevice, s1));
    CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, bytes_vec, cudaMemcpyHostToDevice, s1));
    int block = 256;
    int grid = (Nvec + block - 1) / block;
    vecAdd<<<grid, block, 0, s1>>>(d_a, d_b, d_c, Nvec);
    CUDA_CHECK(cudaMemcpyAsync(h_c, d_c, bytes_vec, cudaMemcpyDeviceToHost, s1));

#ifdef USE_NVTX
    nvtxRangePop();
#endif

    // Matrix multiply: moderate size to exercise SMs
    const int N = 1024; // N x N
    const int matBytes = N * N * sizeof(float);
    float *h_MA = (float*)malloc(matBytes);
    float *h_MB = (float*)malloc(matBytes);
    float *h_MC = (float*)malloc(matBytes);
    for (int i = 0; i < N*N; ++i) { h_MA[i] = 1.0f; h_MB[i] = 1.0f; h_MC[i] = 0.0f; }
    float *d_MA, *d_MB, *d_MC;
    CUDA_CHECK(cudaMalloc((void**)&d_MA, matBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_MB, matBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_MC, matBytes));

#ifdef USE_NVTX
    nvtxRangePushA("MatrixMultiply-Region");
#endif

    // async copy to device in stream s2 to overlap with s1
    CUDA_CHECK(cudaMemcpyAsync(d_MA, h_MA, matBytes, cudaMemcpyHostToDevice, s2));
    CUDA_CHECK(cudaMemcpyAsync(d_MB, h_MB, matBytes, cudaMemcpyHostToDevice, s2));

    dim3 TILE(16,16);
    dim3 blocks((N + TILE.x - 1)/TILE.x, (N + TILE.y - 1)/TILE.y);
    dim3 threads(TILE.x, TILE.y);
    // call templated kernel instantiated with TILE=16
    matMulTiled<16><<<blocks, threads, 0, s2>>>(d_MA, d_MB, d_MC, N);

#ifdef USE_NVTX
    nvtxRangePop();
#endif

    // Reduction (sum) - use a moderate size
    const int Nred = 1<<20;
    float *h_red = (float*)malloc(Nred * sizeof(float));
    for (int i = 0; i < Nred; ++i) h_red[i] = 1.0f;
    float *d_red;
    CUDA_CHECK(cudaMalloc((void**)&d_red, Nred * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(d_red, h_red, Nred * sizeof(float), cudaMemcpyHostToDevice, s1));

#ifdef USE_NVTX
    nvtxRangePushA("Reduction-Region");
#endif

    int threadsR = 256;
    int blocksR = (Nred + threadsR * 2 - 1) / (threadsR * 2);
    float *d_partial;
    CUDA_CHECK(cudaMalloc((void**)&d_partial, blocksR * sizeof(float)));
    blockReduceSum<<<blocksR, threadsR, threadsR * sizeof(float), s1>>>(d_red, d_partial, Nred);

#ifdef USE_NVTX
    nvtxRangePop();
#endif

    // Atomic kernel to create irregular accesses
    int *d_intarr;
    int nInt = 1<<20;
    CUDA_CHECK(cudaMalloc((void**)&d_intarr, nInt * sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(d_intarr, 0, nInt * sizeof(int), s2));
    int threadsA = 256;
    int blocksA = (nInt + threadsA - 1) / threadsA;
    randomAtomicKernel<<<blocksA, threadsA, 0, s2>>>(d_intarr, nInt, 100);

    // synchronize streams and device
    CUDA_CHECK(cudaStreamSynchronize(s1));
    CUDA_CHECK(cudaStreamSynchronize(s2));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Simple validation and timing output
    double sumC = 0.0;
    for (int i = 0; i < Nvec; ++i) sumC += h_c[i];
    printf("VecAdd: sum(h_c) ~ %.1f (expected %f)\n", sumC, (1.0f+2.0f)* (double)Nvec);

    // read a single element of matrix result to validate
    float valC = 0.0f;
    CUDA_CHECK(cudaMemcpy(&valC, d_MC, sizeof(float), cudaMemcpyDeviceToHost));
    printf("matMul C[0] = %f (expected %f)\n", valC, (float)N);

    // reduction partial results copy and finalize on CPU
    float *h_partial = (float*)malloc(blocksR * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, blocksR * sizeof(float), cudaMemcpyDeviceToHost));
    double total = 0.0;
    for (int i = 0; i < blocksR; ++i) total += h_partial[i];
    printf("Reduction sum = %f (expected %f)\n", total, (double)Nred);

    // cleanup
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_MA)); CUDA_CHECK(cudaFree(d_MB)); CUDA_CHECK(cudaFree(d_MC));
    CUDA_CHECK(cudaFree(d_red)); CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_intarr));
    CUDA_CHECK(cudaFreeHost(h_a)); CUDA_CHECK(cudaFreeHost(h_b)); CUDA_CHECK(cudaFreeHost(h_c));
    free(h_MA); free(h_MB); free(h_MC); free(h_red); free(h_partial);

    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaStreamDestroy(s2));

    printf("Done.\n");
    return 0;
}
