#include <stdio.h>
#include <cuda.h>
#include "nccl.h"
#include "mpi.h"

int main(int argc, char *argv[])
{
    int rank, nranks; // 申明进程和进程总数
    int n = 5;
    float *vec;                               // 在host端定义一个向量
    vec = (float *)malloc(n * sizeof(float)); // 在host端分配内存
    for (int i = 0; i < n; i++)
    {
        vec[i] = i;
    }
    float *hostbuff; // 在host端定义另一个向量，后面用来传输cpu-gpu之间的数据
    hostbuff = (float *)malloc(n * sizeof(float));
    MPI_Init(&argc, &argv);                 // 初始化MPI环境
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // 得到当前设备的rank
    MPI_Comm_size(MPI_COMM_WORLD, &nranks); // 得到总进程数，后面希望一个进程管理一个gpu

    cudaSetDevice(rank); // 指定使用哪个显卡，后面的cuda代码会在对应的显卡上执行
    if (rank < n)
    {
        printf("GPU id:%d, rank/nRanks:%d/%d\n", rank, rank, nranks);
    }
    float *devicebuff;
    cudaMalloc((void **)&devicebuff, n * sizeof(float));                         // cuda上分配向量内存
    cudaMemcpy(devicebuff, vec, n * sizeof(float), cudaMemcpyHostToDevice);      // 把host端的vec复制给cuda端的devicebuff
    cudaMemcpy(hostbuff, devicebuff, n * sizeof(float), cudaMemcpyDeviceToHost); // 把cuda端的devicebuff传回cpu端，hostbuff接受数据
    printf("device id:%d\n", rank);
    for (int i = 0; i < n; i++)
    {
        printf("%.2f ", hostbuff[i]); // 检查刚刚cuda端的向量devicebuff是否成功复制vec
    }
    printf("\n");
    ncclUniqueId id; // 申明nccl通信符
    ncclComm_t comm; // 申明nccl通信器
    cudaStream_t s;  // 申明cudastream流，作用不详

    if (rank == 0)
        ncclGetUniqueId(&id);                                        // 初始化通信符，每个GPU都需要初始化
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD); // 从root=0把数据id分发给其他进程
    // ncclGetUniqueId(&id);//这种做法不可行，还是需要rank=0的时候初始化，然后bcast分发给其余进程
    cudaStreamCreate(&s);                      // 创建stream对象，作用不详
    ncclCommInitRank(&comm, nranks, id, rank); // 初始化通信
    //----------------------------------------Reduce测试
    float *recv; // 用来接受所有进程上devicebuff规约以后的数据
    cudaMalloc((void **)&recv, n * sizeof(float));
    ncclReduce((const void *)devicebuff, (void *)recv, n, ncclFloat, ncclSum, 0, comm, s); // 从其他gpu搜集devicebuff，规约到cuda:0，规约方式为sum
    cudaStreamSynchronize(s);                                                              // 同步，保证所有的显卡上数据运输成功
    cudaMemcpy(hostbuff, recv, n * sizeof(float), cudaMemcpyDeviceToHost);                 // 把recv传回cpu，检查结果正确性
    printf("reduce GPU id:%d\n", rank);
    for (int i = 0; i < n; i++)
    {
        printf("%.2f ", hostbuff[i]); // 除了cuda:0以外，其他显卡recv向量元素都是0
    }
    printf("\n");
    //-----------------------SendRecv测试
    if (rank == 0)
    {
        printf("GPU send data from cuda:%d\n", rank);
        ncclSend(recv, n, ncclFloat, nranks - 1, comm, s);
    }
    else if (rank == nranks - 1)
    {
        printf("GPU recv data to cuda:%d\n", rank);
        ncclRecv(recv, n, ncclFloat, 0, comm, s);
    }
    cudaMemcpy(hostbuff, recv, n * sizeof(float), cudaMemcpyDeviceToHost); // 把recv传回cpu，检查结果正确性

    printf("send recv GPU id:%d\n", rank);
    printf("[");
    for (int i = 0; i < n; i++)
    {
        printf("%.2f ", hostbuff[i]); // 除了cuda:0以外，其他显卡recv向量元素都是0
    }
    printf("]");
    printf("\n");
    cudaStreamSynchronize(s); // 同步，保证所有的显卡上数据打印成功
    //--------------------AllReduce测试
    ncclAllReduce((const void *)recv, (void *)devicebuff, n, ncclFloat, ncclAvg, comm, s); // 把所有显卡recv平均规约，然后分发给所有显卡，存在devicebuff上
    cudaStreamSynchronize(s);                                                              // 同步，保证所有的显卡上数据运输成功
    cudaMemcpy(hostbuff, devicebuff, n * sizeof(float), cudaMemcpyDeviceToHost);           // 把devicebuff传回cpu，检查结果正确性
    printf("Allreduce Average GPU id:%d\n", rank);
    printf("[");
    for (int i = 0; i < n; i++)
    {
        printf("%.2f ", hostbuff[i]);
    }
    printf("]");
    printf("\n");

    cudaFree(devicebuff); // 释放内存
    cudaFree(recv);
    ncclCommDestroy(comm); // 销毁通信器
    MPI_Finalize();
    free(vec);
    free(hostbuff);
    return 0;
}
