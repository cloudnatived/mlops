#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include "nccl.h"
#include "mpi.h"
double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
template <int BLOCK_DIM>
__global__ void softmaxKernel(float *input, float *output, int size)
{

    __shared__ float maxData[BLOCK_DIM];
    maxData[threadIdx.x] = -__FLT_MAX__;
    for (int i = threadIdx.x; i < size; i += BLOCK_DIM)
    {
        maxData[threadIdx.x] = max(maxData[threadIdx.x], input[i]);
    }
    for (int strip = blockDim.x / 2; strip > 0; strip /= 2)
    {
        if (threadIdx.x < strip)
        {
            maxData[threadIdx.x] = max(maxData[threadIdx.x], maxData[threadIdx.x + strip]);
        }
        __syncthreads();
    }
    __syncthreads();
    float localMax = maxData[0];

    __shared__ float sumData[BLOCK_DIM];
    sumData[threadIdx.x] = 0;
    for (int i = threadIdx.x; i < size; i += BLOCK_DIM)
    {
        sumData[threadIdx.x] += __expf(input[i] - localMax);
    }
    for (int strip = blockDim.x / 2; strip > 0; strip /= 2)
    {
        if (threadIdx.x < strip)
        {
            sumData[threadIdx.x] += sumData[threadIdx.x + strip];
        }
        __syncthreads();
    }
    __syncthreads();
    float localSum = sumData[0];

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size)
    {
        output[index] = __expf(input[index] - localMax) * __fdividef(1.0F, localSum);
    }
}
void cpuSoftmax(float *cpu_input, float *cpu_output, int size, int rank, int nRanks)
{
    cudaSetDevice(rank); // 指定使用哪个显卡，后面的cuda代码会在对应的显卡上执行
    ncclUniqueId id;     // 申明nccl通信符
    ncclComm_t comm;     // 申明nccl通信器
    cudaStream_t s;      // 申明cudastream流，作用不详

    if (rank == 0)
        ncclGetUniqueId(&id);                                        // 初始化通信符，每个GPU都需要初始化
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD); // 从root=0把数据id分发给其他进程
    // ncclGetUniqueId(&id);//这种做法不可行，还是需要rank=0的时候初始化，然后bcast分发给其余进程
    cudaStreamCreate(&s);                      // 创建stream对象，作用不详
    ncclCommInitRank(&comm, nRanks, id, rank); // 初始化通信

    int remain = size % nRanks;
    int stepEasy = (size - remain) / nRanks;
    int stepHard = stepEasy + 1;
    int step = (rank < remain ? stepHard : stepEasy);
    int indStart = (rank < remain ? rank * stepHard : (remain * stepHard + (rank - remain) * stepEasy));
    //----------------------------------------
    double st, ela;
    st = get_walltime();

    float *input, *output;
    cudaMalloc((void **)&input, step * sizeof(float));

    cudaMalloc((void **)&output, step * sizeof(float));

    cudaMemcpy(input, cpu_input + indStart, step * sizeof(float), cudaMemcpyHostToDevice);
    int BLOCK_DIM = 1024;
    int num_blocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
    dim3 block_dim(BLOCK_DIM, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);

    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    softmaxKernel<1024><<<grid_dim, block_dim>>>(input, output, size);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaStreamSynchronize(s);
    cudaMemcpy(cpu_output + indStart, output, step * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(input);
    cudaFree(output);

    ela = get_walltime() - st;

    printf("GPU:%d, kernel time:%.4f ms, use time:%.4f ms\n", rank, ker_time, ela * 1000);
    ncclCommDestroy(comm); // 销毁通信器
}

int main(int argc, char *argv[])
{
    float *cpu_input, *cpu_output;
    int size = 1600;

    cpu_input = (float *)malloc(size * sizeof(float));
    cpu_output = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        cpu_input[i] = i % 3;
    }
    int rank, nRanks;
    MPI_Init(&argc, &argv);                 // 初始化MPI环境
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // 得到当前设备的rank
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks); // 得到总进程数，后面希望一个进程管理一个gpu

    cpuSoftmax(cpu_input, cpu_output, size, rank, nRanks);
    MPI_Finalize();
    free(cpu_input);
    free(cpu_output);
    return 0;
}
