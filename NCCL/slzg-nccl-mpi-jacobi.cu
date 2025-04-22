#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include "mpi.h"
#include "nccl.h"
const double alpha = 0.5;
const int max_iter = 20000;
double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}

double u_acc(double x, double y)
{
    return (1.0 - pow(x, 2)) * (1.0 - pow(y, 2));
}
double f(double x, double y)
{

    return 2 * (1.0 - pow(x, 2)) + 2 * (1.0 - pow(y, 2)) + alpha * (1.0 - pow(x, 2)) * (1.0 - pow(y, 2));
}
void init_data(double bound[][2], double *u_new, double *u, double *f_1d, double dx, double dy, int M, int N)
{
    int i, j;
    double xx, yy;

    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            xx = bound[0][0] + i * dx;
            yy = bound[1][0] + j * dy;
            if (j == 0 || j == N - 1 || i == 0 || i == M - 1)
            {
                f_1d[i * N + j] = 0;
                u_new[i * N + j] = u_acc(xx, yy);
                u[i * N + j] = u_acc(xx, yy);
            }
            else
            {
                f_1d[i * N + j] = f(xx, yy) * (dx * dx + dy * dy);
                u_new[i * N + j] = 0;
                u[i * N + j] = u_acc(xx, yy);
            }
        }
    }
}
double L1_err(double *u_new, double *u, int N, int indStart, int step)
{
    int i, j;
    double err = 0;
    for (i = indStart; i < indStart + step; i++)
    {
        for (j = 0; j < N; j++)
        {
            err = fmax(err, fabs(u_new[i * N + j] - u[i * N + j]));
        }
    }
    return err;
}

__global__ void
cuda_update(double *cuda_u_old, double *cuda_u_new, int N, int indStart, int step)
{
    // must void type
    int i = indStart + threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (j < N && i < indStart + step)
    {
        cuda_u_old[i * N + j] = cuda_u_new[i * N + j];
    }
}
__global__ void cuda_Jacobi(double *cuda_f_1d, double *cuda_u_old, double *cuda_u_new, double r1, double r2, double r3, double r, int M, int N, int indStart, int step)
{
    // must void type
    int i = indStart + threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    double resid = 0;
    if (i < indStart + step)
    {
        if (j > 0 && j < N - 1 && i > 0 && i < M - 1)
        {
            resid = cuda_f_1d[i * N + j] -
                    (r1 * (cuda_u_old[(i - 1) * N + j - 1] + cuda_u_old[(i - 1) * N + j + 1]) +
                     r3 * (cuda_u_old[(i - 1) * N + j] + cuda_u_old[(i + 1) * N + j]) +
                     r1 * (cuda_u_old[(i + 1) * N + j - 1] + cuda_u_old[(i + 1) * N + j + 1]) +
                     r2 * (cuda_u_old[i * N + j - 1] + cuda_u_old[i * N + j + 1]) + r * cuda_u_old[i * N + j]);
            cuda_u_new[i * N + j] = cuda_u_old[i * N + j] + resid / r;
        }
    }
}

void cuda_solve(double *f_1d, double *u_old, double *u_new, double r1, double r2, double r3, double r, int M, int N, int rank, int nranks)
{

    int indStart, step;
    cudaSetDevice(rank); // 指定使用哪个显卡，后面的cuda代码会在对应的显卡上执行
    ncclUniqueId id;     // 申明nccl通信符
    ncclComm_t comm;     // 申明nccl通信器
    cudaStream_t s;      // 申明cudastream流，作用不详

    if (rank == 0)
        ncclGetUniqueId(&id);                                        // 初始化通信符，每个GPU都需要初始化
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD); // 从root=0把数据id分发给其他进程
    // ncclGetUniqueId(&id);//这种做法不可行，还是需要rank=0的时候初始化，然后bcast分发给其余进程
    cudaStreamCreate(&s);                      // 创建stream对象，作用不详
    ncclCommInitRank(&comm, nranks, id, rank); // 初始化通信

    double *cuda_f_1d, *cuda_u_old, *cuda_u_new;
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //-------------------

    if (nranks > 1)
    {
        int remain = M % nranks;
        int stepEasy = (M - remain) / nranks;
        int stepHard = stepEasy + 1;
        step = (rank < remain ? stepHard : stepEasy);
        indStart = (rank < remain ? (rank * stepHard) : (remain * stepHard + (rank - remain) * stepEasy));
    }
    else
    {
        indStart = 0;
        step = M;
    }
    cudaMalloc((void **)&cuda_f_1d, M * N * sizeof(double));
    cudaMalloc((void **)&cuda_u_old, M * N * sizeof(double));
    cudaMalloc((void **)&cuda_u_new, M * N * sizeof(double));
    if (nranks == 1)
    {
        cudaMemcpy(cuda_f_1d + indStart * N, f_1d + indStart * N, step * N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_u_old + indStart * N, u_old + indStart * N, step * N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_u_new + indStart * N, u_new + indStart * N, step * N * sizeof(double), cudaMemcpyHostToDevice);
    }
    else
    {
        cudaMemcpyAsync(cuda_f_1d + indStart * N, f_1d + indStart * N, step * N * sizeof(double), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(cuda_u_old + indStart * N, u_old + indStart * N, step * N * sizeof(double), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(cuda_u_new + indStart * N, u_new + indStart * N, step * N * sizeof(double), cudaMemcpyHostToDevice, s);
    }

    int BLOCK_DIM_x = 32;
    int BLOCK_DIM_y = 32;
    int num_blocks_x = (step + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    int num_blocks_y = (N + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);

    int k = 0;
    double updateStart, updateEnd, updateTime;
    double commStart, commEnd, commTime;
    double jacobiStart, jacobiEnd, jacobiTime;
    updateTime = 0;
    commTime = 0;
    jacobiTime = 0;
    cudaEventRecord(start, 0);
    if (nranks > 1)
    {
        while (k < max_iter)
        {
            updateStart = get_walltime();
            cuda_update<<<grid_dim, block_dim, 0, s>>>(cuda_u_old, cuda_u_new, N, indStart, step);
            cudaStreamSynchronize(s);
            updateEnd = get_walltime() - updateStart;
            commStart = get_walltime();
            if (rank == 0)
            {
                ncclSend(cuda_u_old + indStart * N + (step - 1) * N, N, ncclDouble, rank + 1, comm, s);
                ncclRecv(cuda_u_old + indStart * N + step * N, N, ncclDouble, rank + 1, comm, s);
            }
            else if (rank > 0 && rank < nranks - 1)
            {
                ncclRecv(cuda_u_old + indStart * N - N, N, ncclDouble, rank - 1, comm, s);
                ncclSend(cuda_u_old + indStart * N, N, ncclDouble, rank - 1, comm, s);

                ncclSend(cuda_u_old + indStart * N + (step - 1) * N, N, ncclDouble, rank + 1, comm, s);
                ncclRecv(cuda_u_old + indStart * N + step * N, N, ncclDouble, rank + 1, comm, s);
            }
            else
            {
                ncclRecv(cuda_u_old + indStart * N - N, N, ncclDouble, rank - 1, comm, s);
                ncclSend(cuda_u_old + indStart * N, N, ncclDouble, rank - 1, comm, s);
            }

            cudaStreamSynchronize(s);
            commEnd = get_walltime() - commStart;
            jacobiStart = get_walltime();
            cuda_Jacobi<<<grid_dim, block_dim, 0, s>>>(cuda_f_1d, cuda_u_old, cuda_u_new, r1, r2, r3, r, M, N, indStart, step);
            jacobiEnd = get_walltime() - jacobiStart;

            updateTime += updateEnd;
            commTime += commEnd;
            jacobiTime += jacobiEnd;
            k += 1;
        }
    }
    else
    {
        while (k < max_iter)
        {
            updateStart = get_walltime();
            cuda_update<<<grid_dim, block_dim>>>(cuda_u_old, cuda_u_new, N, indStart, step);
            cudaDeviceSynchronize(); // must wait
            updateEnd = get_walltime() - updateStart;

            jacobiStart = get_walltime();
            cuda_Jacobi<<<grid_dim, block_dim>>>(cuda_f_1d, cuda_u_old, cuda_u_new, r1, r2, r3, r, M, N, indStart, step);
            jacobiEnd = get_walltime() - jacobiStart;

            updateTime += updateEnd;
            jacobiTime += jacobiEnd;

            k += 1;
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time

    cudaMemcpy(f_1d + indStart * N, cuda_f_1d + indStart * N, step * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(u_old + indStart * N, cuda_u_old + indStart * N, step * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(u_new + indStart * N, cuda_u_new + indStart * N, step * N * sizeof(double), cudaMemcpyDeviceToHost);

    ncclCommDestroy(comm); // 销毁通信器
    cudaFree(cuda_f_1d);
    cudaFree(cuda_u_old);
    cudaFree(cuda_u_new);

    if (rank == 0)
    {
        printf("grid dim: %d, %d\n", grid_dim.x, grid_dim.y);
        printf("block dim: %d, %d\n", block_dim.x, block_dim.y);
        printf("update:%.5f, communication:%.5f, jacobi:%.5f, kernel launch time:%.5f\n", updateTime, commTime, jacobiTime, ker_time / 1000.);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char *argv[])
{
    int rank, nranks;
    MPI_Init(&argc, &argv);                 // 初始化MPI环境
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // 得到当前设备的rank
    MPI_Comm_size(MPI_COMM_WORLD, &nranks); // 得到总进程数，后面希望一个进程管理一个gpu
    int M = 64;                             // default value
    int N = 128;
    double *u_old, *u_new, *f_1d, *u;
    if (nranks == 1)
    {
        u_old = (double *)malloc(M * N * sizeof(double));
        u_new = (double *)malloc(M * N * sizeof(double));
        f_1d = (double *)malloc(M * N * sizeof(double));
        u = (double *)malloc(M * N * sizeof(double));
    }
    else
    {
        cudaHostAlloc((void **)&u_old, M * N * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc((void **)&u_new, M * N * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc((void **)&f_1d, M * N * sizeof(double), cudaHostAllocDefault);
        cudaHostAlloc((void **)&u, M * N * sizeof(double), cudaHostAllocDefault);
    }
    double bound[2][2] = {{-1, 1}, {-1, 1}};
    double dx, dy;

    if (rank == 0)
    {
        printf("the freedom (M,N) = (%d,%d),max_iter = %d,--------------------\n", M, N, max_iter);
    }

    dx = (bound[0][1] - bound[0][0]) / (M - 1);
    dy = (bound[1][1] - bound[1][0]) / (N - 1);

    double r1 = -0.5, r2 = -pow(dx / dy, 2);
    double r3 = -pow(dy / dx, 2), r = 2 * (1 - r2 - r3) + alpha * (dx * dx + dy * dy);

    double st, ela;

    init_data(bound, u_new, u, f_1d, dx, dy, M, N);

    st = get_walltime();
    cuda_solve(f_1d, u_old, u_new, r1, r2, r3, r, M, N, rank, nranks);
    ela = get_walltime() - st;

    double err = 0;
    int indStart, step;
    if (nranks > 1)
    {
        int remain = M % nranks;
        int stepEasy = (M - remain) / nranks;
        int stepHard = stepEasy + 1;
        step = (rank < remain ? stepHard : stepEasy);
        indStart = (rank < remain ? (rank * stepHard) : (remain * stepHard + (rank - remain) * stepEasy));
    }
    else
    {
        indStart = 0;
        step = M;
    }
    err = L1_err(u_new, u, N, indStart, step);
    double myerr;
    MPI_Allreduce(&err, &myerr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0)
    {
        printf("GPU: use time:%.2f,the L1_err :%.4e\n", ela, myerr);
    }
    if (nranks == 1)
    {
        free(u_old);
        free(u_new);
        free(f_1d);
        free(u);
    }
    else
    {
        cudaFreeHost(u_old);
        cudaFreeHost(u_new);
        cudaFreeHost(f_1d);
        cudaFreeHost(u);
    }
    MPI_Finalize();
    return 0;
}
