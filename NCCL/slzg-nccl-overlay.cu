#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
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
__global__ void cuda_JacobiB(double *cuda_f_1d, double *cuda_u_old, double *cuda_u_new, double r1, double r2, double r3, double r, int M, int N, int indStart, int step)
{
    // must void type
    int i = indStart + threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    double resid = 0;
    if (i == indStart || i == indStart + step - 1)
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
__global__ void cuda_JacobiI(double *cuda_f_1d, double *cuda_u_old, double *cuda_u_new, double r1, double r2, double r3, double r, int M, int N, int indStart, int step)
{
    // must void type
    int i = indStart + 1 + threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    double resid = 0;
    if (j > 0 && j < N - 1 && i < indStart + step - 1)
    {
        resid = cuda_f_1d[i * N + j] -
                (r1 * (cuda_u_old[(i - 1) * N + j - 1] + cuda_u_old[(i - 1) * N + j + 1]) +
                 r3 * (cuda_u_old[(i - 1) * N + j] + cuda_u_old[(i + 1) * N + j]) +
                 r1 * (cuda_u_old[(i + 1) * N + j - 1] + cuda_u_old[(i + 1) * N + j + 1]) +
                 r2 * (cuda_u_old[i * N + j - 1] + cuda_u_old[i * N + j + 1]) + r * cuda_u_old[i * N + j]);
        cuda_u_new[i * N + j] = cuda_u_old[i * N + j] + resid / r;
    }
}
void cuda_solve(double *f_1d, double *u_old, double *u_new, double *u, double r1, double r2, double r3, double r, int M, int N, int nDev)
{
    int *devs = (int *)malloc(sizeof(int) * nDev);
    for (int i = 0; i < nDev; i++)
    {
        devs[i] = i;
    }
    ncclComm_t *comms = (ncclComm_t *)malloc(sizeof(ncclComm_t) * nDev);
    cudaStream_t *s = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nDev);
    int *indStarts = (int *)malloc(sizeof(int) * nDev);
    int *steps = (int *)malloc(sizeof(int) * nDev);
    int remain = M % nDev;
    int stepEasy = (M - remain) / nDev;
    int stepHard = stepEasy + 1;
    for (int i = 0; i < nDev; i++)
    {
        if (i < remain)
        {
            steps[i] = stepHard;
            indStarts[i] = i * stepHard;
        }
        else
        {
            steps[i] = stepEasy;
            indStarts[i] = remain * stepHard + (i - remain) * stepEasy;
        }
    }
    if (nDev == 1)
    {
        steps[0] = M;
        indStarts[0] = 0;
    }
    double **cuda_f_1d = (double **)malloc(nDev * sizeof(double *));
    double **cuda_u_old = (double **)malloc(nDev * sizeof(double *));
    double **cuda_u_new = (double **)malloc(nDev * sizeof(double *));
    for (int i = 0; i < nDev; i++)
    {
        cudaSetDevice(i);
        cudaMalloc((void **)&cuda_f_1d[i], M * N * sizeof(double));
        cudaMalloc((void **)&cuda_u_old[i], M * N * sizeof(double));
        cudaMalloc((void **)&cuda_u_new[i], M * N * sizeof(double));
        cudaMemcpy(cuda_f_1d[i] + indStarts[i] * N, f_1d + indStarts[i] * N, steps[i] * N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_u_old[i] + indStarts[i] * N, u_old + indStarts[i] * N, steps[i] * N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_u_new[i] + indStarts[i] * N, u_new + indStarts[i] * N, steps[i] * N * sizeof(double), cudaMemcpyHostToDevice);
        cudaStreamCreate(s + i);
    }
    ncclCommInitAll(comms, nDev, devs);
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //-------------------
    int BLOCK_DIM_x = 32;
    int BLOCK_DIM_y = 32;
    int num_blocks_x;
    int num_blocks_y;
    dim3 block_dim;
    dim3 grid_dim;
    int k = 0;
    double updateStart, updateEnd, updateTime;
    double commStart, commEnd, commTime;
    double jacobiStart, jacobiEnd, jacobiTime;
    updateTime = 0;
    commTime = 0;
    jacobiTime = 0;
    cudaEventRecord(start, 0);
    if (nDev == 1)
    {
        int i = 0;
        block_dim.x = BLOCK_DIM_x;
        block_dim.y = BLOCK_DIM_y;
        num_blocks_x = (steps[i] + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        num_blocks_y = (N + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        grid_dim.x = num_blocks_x;
        grid_dim.y = num_blocks_y;
        while (k < max_iter)
        {
            updateStart = get_walltime();
            cuda_update<<<grid_dim, block_dim>>>(cuda_u_old[i], cuda_u_new[i], N, indStarts[i], steps[i]);
            cudaDeviceSynchronize();
            updateEnd = get_walltime() - updateStart;

            jacobiStart = get_walltime();
            cuda_Jacobi<<<grid_dim, block_dim>>>(cuda_f_1d[i], cuda_u_old[i], cuda_u_new[i], r1, r2, r3, r, M, N, indStarts[i], steps[i]);
            cudaDeviceSynchronize();
            jacobiEnd = get_walltime() - jacobiStart;

            updateTime += updateEnd;
            jacobiTime += jacobiEnd;

            k += 1;
        }
    }
    else
    {
        while (k < max_iter)
        {
            updateStart = get_walltime();
            for (int i = 0; i < nDev; i++)
            {
                cudaSetDevice(i);
                block_dim.x = BLOCK_DIM_x;
                block_dim.y = BLOCK_DIM_y;
                num_blocks_x = (steps[i] + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
                num_blocks_y = (N + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
                grid_dim.x = num_blocks_x;
                grid_dim.y = num_blocks_y;

                cuda_update<<<grid_dim, block_dim, 0, s[i]>>>(cuda_u_old[i], cuda_u_new[i], N, indStarts[i], steps[i]);
                cudaStreamSynchronize(s[i]);
            }
            updateEnd = get_walltime() - updateStart;
            ncclGroupStart();
            commStart = get_walltime();
            for (int i = 0; i < nDev; i++)
            {
                cudaSetDevice(i);

                if (i == 0)
                {
                    ncclSend(cuda_u_old[i] + indStarts[i] * N + (steps[i] - 1) * N, N, ncclDouble, i + 1, comms[i], s[i]);
                    ncclRecv(cuda_u_old[i] + indStarts[i] * N + steps[i] * N, N, ncclDouble, i + 1, comms[i], s[i]);
                }
                else if (i > 0 && i < nDev - 1)
                {
                    ncclRecv(cuda_u_old[i] + indStarts[i] * N - N, N, ncclDouble, i - 1, comms[i], s[i]);
                    ncclSend(cuda_u_old[i] + indStarts[i] * N, N, ncclDouble, i - 1, comms[i], s[i]);

                    ncclSend(cuda_u_old[i] + indStarts[i] * N + (steps[i] - 1) * N, N, ncclDouble, i + 1, comms[i], s[i]);
                    ncclRecv(cuda_u_old[i] + indStarts[i] * N + steps[i] * N, N, ncclDouble, i + 1, comms[i], s[i]);
                }
                else
                {
                    ncclRecv(cuda_u_old[i] + indStarts[i] * N - N, N, ncclDouble, i - 1, comms[i], s[i]);
                    ncclSend(cuda_u_old[i] + indStarts[i] * N, N, ncclDouble, i - 1, comms[i], s[i]);
                }

                // cudaStreamSynchronize(s[i]);
            }
            ncclGroupEnd();
            commEnd = get_walltime() - commStart;
            jacobiStart = get_walltime();
            for (int i = 0; i < nDev; i++)
            {
                cudaSetDevice(i);
                block_dim.x = BLOCK_DIM_x;
                block_dim.y = BLOCK_DIM_y;
                num_blocks_x = (steps[i] + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
                num_blocks_y = (N + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
                grid_dim.x = num_blocks_x;
                grid_dim.y = num_blocks_y;
                cuda_JacobiI<<<grid_dim, block_dim, 0, s[i]>>>(cuda_f_1d[i], cuda_u_old[i], cuda_u_new[i], r1, r2, r3, r, M, N, indStarts[i], steps[i]);
                cuda_JacobiB<<<grid_dim, block_dim, 0, s[i]>>>(cuda_f_1d[i], cuda_u_old[i], cuda_u_new[i], r1, r2, r3, r, M, N, indStarts[i], steps[i]);
                cudaStreamSynchronize(s[i]);
            }
            jacobiEnd = get_walltime() - jacobiStart;
            updateTime += updateEnd;
            commTime += commEnd;
            jacobiTime += jacobiEnd;
            k += 1;
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    for (int i = 0; i < nDev; i++)
    {
        cudaSetDevice(i);
        cudaStreamSynchronize(s[i]);
        cudaMemcpy(f_1d + indStarts[i] * N, cuda_f_1d[i] + indStarts[i] * N, steps[i] * N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(u_old + indStarts[i] * N, cuda_u_old[i] + indStarts[i] * N, steps[i] * N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(u_new + indStarts[i] * N, cuda_u_new[i] + indStarts[i] * N, steps[i] * N * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(cuda_f_1d[i]);
        cudaFree(cuda_u_old[i]);
        cudaFree(cuda_u_new[i]);

        if (i == 0)
        {

            printf("grid dim: %d, %d\n", grid_dim.x, grid_dim.y);
            printf("block dim: %d, %d\n", block_dim.x, block_dim.y);
            printf("update:%.5f, communication:%.5f, jacobi:%.5f, kernel launch time:%.5f\n", updateTime, commTime, jacobiTime, ker_time / 1000.);
        }
    }
    double err = 0;

    err = L1_err(u_new, u, N, 0, M);
    printf("error:%.4e\n", err);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for (int i = 0; i < nDev; ++i)
        ncclCommDestroy(comms[i]);
    free(devs);
    free(indStarts);
    free(steps);
}

int main(int argc, char **argv)
{
    int nDev = 4;
    if (argc > 1)
        nDev = atoi(argv[1]); // user-specified value
    int M = 64;               // default value
    int N = 128;
    double *u_old, *u_new, *f_1d, *u;
    u_old = (double *)malloc(M * N * sizeof(double));
    u_new = (double *)malloc(M * N * sizeof(double));
    f_1d = (double *)malloc(M * N * sizeof(double));
    u = (double *)malloc(M * N * sizeof(double));
    double bound[2][2] = {{-1, 1}, {-1, 1}};
    double dx, dy;

    printf("device num:%d, the freedom (M,N) = (%d,%d),max_iter = %d--------------------\n", nDev, M, N, max_iter);

    dx = (bound[0][1] - bound[0][0]) / (M - 1);
    dy = (bound[1][1] - bound[1][0]) / (N - 1);

    double r1 = -0.5, r2 = -pow(dx / dy, 2);
    double r3 = -pow(dy / dx, 2), r = 2 * (1 - r2 - r3) + alpha * (dx * dx + dy * dy);

    double st, ela;

    init_data(bound, u_new, u, f_1d, dx, dy, M, N);

    st = get_walltime();
    cuda_solve(f_1d, u_old, u_new, u, r1, r2, r3, r, M, N, nDev);
    ela = get_walltime() - st;

    printf("GPU: use time:%.2f\n", ela);
    free(u_old);
    free(u_new);
    free(f_1d);
    free(u);

    return 0;
}
