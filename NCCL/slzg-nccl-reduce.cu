#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"

#define CUDACHECK(cmd)                                           \
    do                                                           \
    {                                                            \
        cudaError_t err = cmd;                                   \
        if (err != cudaSuccess)                                  \
        {                                                        \
            printf("Failed: Cuda error %s:%d '%s'\n",            \
                   __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

#define NCCLCHECK(cmd)                                           \
    do                                                           \
    {                                                            \
        ncclResult_t res = cmd;                                  \
        if (res != ncclSuccess)                                  \
        {                                                        \
            printf("Failed, NCCL error %s:%d '%s'\n",            \
                   __FILE__, __LINE__, ncclGetErrorString(res)); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

int main(int argc, char *argv[])
{
    ncclComm_t comms[4];

    // managing 4 devices
    int nDev = 4;
    int size = 32 * 1024 * 1024;
    int devs[4] = {0, 1, 2, 3};

    // allocating and initializing device buffers
    float **sendbuff = (float **)malloc(nDev * sizeof(float *)); // sendbuff是一个长度为nDev的数组，但是sendbuff[i]是指针
    float **recvbuff = (float **)malloc(nDev * sizeof(float *));
    cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nDev);
    float *hostbuff; // 后面用于存储不同device上的recvbuff结果
    hostbuff = (float *)malloc(nDev * size * sizeof(float));
    float *hostsend; // 初始化不同device上的sendbuff
    hostsend = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        hostsend[i] = i;
    }
    for (int i = 0; i < nDev; ++i)
    {
        CUDACHECK(cudaSetDevice(i)); // 后面所有跟CUDA相关的函数都会受到影响
        CUDACHECK(cudaMalloc((void **)sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc((void **)recvbuff + i, size * sizeof(float)));
        // CUDACHECK(cudaMalloc((void **)&sendbuff[i], size * sizeof(float)));
        // CUDACHECK(cudaMalloc((void **)&recvbuff[i], size * sizeof(float)));
        CUDACHECK(cudaMemcpy(sendbuff[i], hostsend, size * sizeof(float), cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float))); // 不管value是什么，cudaMemSet只能初始化数值为0
        cudaMemcpy(hostbuff + i * size, sendbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost);
        printf("host:%.2f\n", hostbuff[i * size + 10]); // 不能直接打印sendbuff[i][10]，因为sendbuff[i]在device上
        CUDACHECK(cudaStreamCreate(s + i));
    }

    // initializing NCCL
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

    // calling NCCL communication API. Group API is required when using
    // multiple devices per thread
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i)
    {
        NCCLCHECK(ncclAllReduce(sendbuff[i], recvbuff[i], size, ncclFloat, ncclSum,
                                comms[i], s[i]));
    }

    NCCLCHECK(ncclGroupEnd());

    // synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i)
    {
        // CUDACHECK(cudaSetDevice(i));//作用范围是后面所有使用cuda相关函数，直到重新设置device编号

        CUDACHECK(cudaStreamSynchronize(s[i]));
        cudaMemcpy(hostbuff + i * size, recvbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost);
        printf("reduceSum host:%.2f\n", hostbuff[i * size + 10]);
    }

    // free device buffers
    for (int i = 0; i < nDev; ++i)
    {
        // CUDACHECK(cudaSetDevice(i));

        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }

    // finalizing NCCL
    for (int i = 0; i < nDev; ++i)
        ncclCommDestroy(comms[i]);
    free(hostbuff);
    free(hostsend);
    printf("Success \n");
    return 0;
}
