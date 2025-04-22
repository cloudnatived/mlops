#include <stdio.h>
#include <cuda_runtime.h>

// CUDA 核函数
__global__ void add(int *a, int *b, int *c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  c[idx] = a[idx] + b[idx];
}

int main() {
  // 定义数据大小
  int N = 1024;

  // 分配主机内存
  int *a, *b, *c;
  a = (int *)malloc(N * sizeof(int));
  b = (int *)malloc(N * sizeof(int));
  c = (int *)malloc(N * sizeof(int));

  // 初始化数据
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i * 2;
  }

  // 分配设备内存
  int *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, N * sizeof(int));
  cudaMalloc((void **)&d_b, N * sizeof(int));
  cudaMalloc((void **)&d_c, N * sizeof(int));

  // 将数据从主机复制到设备
  cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

  // 启动 CUDA 核函数
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(d_a, d_b, d_c);

  // 将结果从设备复制到主机
  cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

  // 打印结果
  for (int i = 0; i < N; i++) {
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }

  // 释放内存
  free(a);
  free(b);
  free(c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}

