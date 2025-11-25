// CUDA_matrix_add_profiling.cu
// nvcc -o CUDA_matrix_add_profiling CUDA_matrix_add_profiling.cu -O3 -lineinfo -lnvToolsExt
// 指定 GPU 架构，且指定正确的 GPU 架构
// nvcc -o CUDA_matrix_add_profiling CUDA_matrix_add_profiling.cu -O3 -gencode arch=compute_70,code=sm_70 --generate-line-info -lnvToolsExt

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "nvtx3/nvToolsExt.h" // NVTX 库

// =======================================================================
// CUDA 错误检查宏
// =======================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// =======================================================================
// CUDA Kernel: C = A + bias
// =======================================================================
__global__ void matrixAddWithBias(float* C, const float* A, float bias, int width, int height)
{
    // 使用2D网格和2D块计算全局线程索引
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查，处理任意矩阵大小
    if (col < width && row < height)
    {
        int idx = row * width + col;
        C[idx] = A[idx] + bias;
    }
}

// =======================================================================
// CPU 验证函数
// =======================================================================
void verifyOnCPU(float* h_C_cpu, const float* h_A, float bias, size_t numElements)
{
    for (size_t i = 0; i < numElements; ++i)
    {
        h_C_cpu[i] = h_A[i] + bias;
    }
}

// =======================================================================
// 验证结果函数
// =======================================================================
bool checkResult(const float* gpuResult, const float* cpuResult, size_t numElements)
{
    const float epsilon = 1e-5f;
    bool success = true;
    for (size_t i = 0; i < numElements; ++i)
    {
        if (fabs(gpuResult[i] - cpuResult[i]) > epsilon)
        {
            fprintf(stderr, "Verification FAILED at index %zu! GPU: %f, CPU: %f\n",
                    i, gpuResult[i], cpuResult[i]);
            success = false;
            return success; // 提前退出
        }
    }
    return success;
}


// =======================================================================
// 测试封装 1: 同步测试 (用于基准测试和 ncu 分析)
// =======================================================================
void run_sync_test(int width, int height, dim3 dimBlock, float bias, const char* testName)
{
    printf("\n--- Running Sync Test: %s ---\n", testName);
    nvtxRangePushA(testName); // NVTX: 开始测试

    // --- 1. 分配和初始化主机内存 ---
    size_t numElements = (size_t)width * height;
    size_t size = numElements * sizeof(float);
    float *h_A, *h_C_gpu, *h_C_cpu;

    h_A = (float*)malloc(size);
    h_C_gpu = (float*)malloc(size);
    h_C_cpu = (float*)malloc(size);
    if (!h_A || !h_C_gpu || !h_C_cpu) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        exit(1);
    }

    for (size_t i = 0; i < numElements; ++i) {
        h_A[i] = (float)i;
    }

    // --- 2. CPU 计算 (用于验证) ---
    verifyOnCPU(h_C_cpu, h_A, bias, numElements);

    // --- 3. 分配设备内存 ---
    float *d_A, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // --- 4. H2D 内存拷贝 ---
    nvtxRangePushA("H2D Copy");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    nvtxRangePop();

    // --- 5. 配置 Kernel 启动参数 ---
    dim3 dimGrid;
    dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;
    printf("Grid: (%u, %u), Block: (%u, %u)\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

    // --- 6. 启动 Kernel ---
    nvtxRangePushA("Kernel: matrixAddWithBias");
    matrixAddWithBias<<<dimGrid, dimBlock>>>(d_C, d_A, bias, width, height);
    CUDA_CHECK(cudaGetLastError()); // 检查内核启动错误
    CUDA_CHECK(cudaDeviceSynchronize()); // 同步以确保内核完成
    nvtxRangePop();

    // --- 7. D2H 内存拷贝 ---
    nvtxRangePushA("D2H Copy");
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));
    nvtxRangePop();

    // --- 8. 验证结果 ---
    printf("Verifying result... ");
    bool success = checkResult(h_C_gpu, h_C_cpu, numElements);
    printf(success ? "SUCCESS!\n" : "FAILED!\n");

    // --- 9. 清理 ---
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C_gpu);
    free(h_C_cpu);

    nvtxRangePop(); // NVTX: 结束测试
}

// =======================================================================
// 测试封装 2: 异步多流测试 (用于 nsys 分析)
// =======================================================================
void run_async_test(int w1, int h1, float b1, int w2, int h2, float b2)
{
    const char* testName = "Multi-Stream Async Test";
    printf("\n--- Running Async Test: %s ---\n", testName);
    nvtxRangePushA(testName);

    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // --- 定义两个不同的问题 ---
    size_t numElem1 = (size_t)w1 * h1;
    size_t size1 = numElem1 * sizeof(float);
    size_t numElem2 = (size_t)w2 * h2;
    size_t size2 = numElem2 * sizeof(float);

    dim3 block1(16, 16);
    dim3 grid1((w1 + block1.x - 1) / block1.x, (h1 + block1.y - 1) / block1.y);
    dim3 block2(32, 8);
    dim3 grid2((w2 + block2.x - 1) / block2.x, (h2 + block2.y - 1) / block2.y);

    // --- 1. 分配 **固定主机内存 (Pinned Memory)** ---
    // 这是实现 H2D/D2H 异步拷贝所必需的
    float *h_A1_pinned, *h_C_gpu1_pinned, *h_C_cpu1;
    float *h_A2_pinned, *h_C_gpu2_pinned, *h_C_cpu2;

    CUDA_CHECK(cudaHostAlloc(&h_A1_pinned, size1, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_C_gpu1_pinned, size1, cudaHostAllocDefault));
    h_C_cpu1 = (float*)malloc(size1);

    CUDA_CHECK(cudaHostAlloc(&h_A2_pinned, size2, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_C_gpu2_pinned, size2, cudaHostAllocDefault));
    h_C_cpu2 = (float*)malloc(size2);

    // --- 2. 初始化主机数据 ---
    for (size_t i = 0; i < numElem1; ++i) h_A1_pinned[i] = (float)i;
    for (size_t i = 0; i < numElem2; ++i) h_A2_pinned[i] = (float)i * 2.0f;

    // --- 3. CPU 计算 (用于验证) ---
    verifyOnCPU(h_C_cpu1, h_A1_pinned, b1, numElem1);
    verifyOnCPU(h_C_cpu2, h_A2_pinned, b2, numElem2);

    // --- 4. 分配设备内存 ---
    float *d_A1, *d_C1, *d_A2, *d_C2;
    CUDA_CHECK(cudaMalloc(&d_A1, size1));
    CUDA_CHECK(cudaMalloc(&d_C1, size1));
    CUDA_CHECK(cudaMalloc(&d_A2, size2));
    CUDA_CHECK(cudaMalloc(&d_C2, size2));

    // --- 5. 在流上注册 NVTX 范围 (用于在 Nsight Systems 中按流显示) ---
    // (注意: `nvtxRangePushA` 已经可以被 nsys 很好地关联到流)
    
    // --- 6. 异步执行 Stream 1 ---
    nvtxRangePushA("Stream 1 Ops");
    CUDA_CHECK(cudaMemcpyAsync(d_A1, h_A1_pinned, size1, cudaMemcpyHostToDevice, stream1));
    matrixAddWithBias<<<grid1, block1, 0, stream1>>>(d_C1, d_A1, b1, w1, h1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_C_gpu1_pinned, d_C1, size1, cudaMemcpyDeviceToHost, stream1));
    nvtxRangePop();

    // --- 7. 异步执行 Stream 2 ---
    nvtxRangePushA("Stream 2 Ops");
    CUDA_CHECK(cudaMemcpyAsync(d_A2, h_A2_pinned, size2, cudaMemcpyHostToDevice, stream2));
    matrixAddWithBias<<<grid2, block2, 0, stream2>>>(d_C2, d_A2, b2, w2, h2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_C_gpu2_pinned, d_C2, size2, cudaMemcpyDeviceToHost, stream2));
    nvtxRangePop();

    // --- 8. 同步所有流 ---
    nvtxRangePushA("Sync All Streams");
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    // --- 9. 验证结果 ---
    printf("Verifying result for Stream 1... ");
    bool s1_success = checkResult(h_C_gpu1_pinned, h_C_cpu1, numElem1);
    printf(s1_success ? "SUCCESS!\n" : "FAILED!\n");

    printf("Verifying result for Stream 2... ");
    bool s2_success = checkResult(h_C_gpu2_pinned, h_C_cpu2, numElem2);
    printf(s2_success ? "SUCCESS!\n" : "FAILED!\n");

    // --- 10. 清理 ---
    cudaFree(d_A1);
    cudaFree(d_C1);
    cudaFree(d_A2);
    cudaFree(d_C2);
    CUDA_CHECK(cudaFreeHost(h_A1_pinned));
    CUDA_CHECK(cudaFreeHost(h_C_gpu1_pinned));
    CUDA_CHECK(cudaFreeHost(h_A2_pinned));
    CUDA_CHECK(cudaFreeHost(h_C_gpu2_pinned));
    free(h_C_cpu1);
    free(h_C_cpu2);
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));

    nvtxRangePop(); // NVTX: 结束测试
}


// =======================================================================
// 主函数
// =======================================================================
int main()
{
    int deviceId;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    printf("Running on device: %s\n", prop.name);

    // --- 运行一系列同步测试 ---
    // (用于 Nsight Compute 的内核级分析)

    // 测试 1: 基准测试 (4K x 4K, 16x16 block)
    run_sync_test(4096, 4096, dim3(16, 16), 10.0f, "Test_4K_16x16");

    // 测试 2: 不同的块大小 (4K x 4K, 32x8 block)
    run_sync_test(4096, 4096, dim3(32, 8), 20.0f, "Test_4K_32x8");

    // 测试 3: 非方形/大矩阵 (8K x 4K, 16x16 block)
    run_sync_test(8192, 4096, dim3(16, 16), 30.0f, "Test_Large_8Kx4K_16x16");

    // 测试 4: 奇数大小 (测试边界检查) (4001 x 3999, 16x16 block)
    run_sync_test(4001, 3999, dim3(16, 16), 40.0f, "Test_OddSize_4001x3999_16x16");

    // --- 运行异步多流测试 ---
    // (用于 Nsight Systems 的系统级分析)
    run_async_test(4096, 4096, 50.0f, // 问题 1
                   2048, 8192, 60.0f  // 问题 2
                  );

    printf("\nAll tests completed.\n");
    return 0;
}
