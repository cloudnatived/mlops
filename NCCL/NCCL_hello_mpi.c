// NCCL_hello_mpi.c
// check env in terminal: mpicc --version; mpirun --version
// 检查库：ldconfig -p | grep libmpi
// apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev
// gcc NCCL_hello_mpi.c -o NCCL_hello_mpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib -Wl,-rpath,/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lm


// 强烈推荐使用方案3（mpicc），因为它是MPI标准提供的编译器包装器，会自动处理所有必要的包含路径和库链接。
// apt-get install openmpi-bin libopenmpi-dev
// mpicc NCCL_hello_mpi.c -o NCCL_hello_mpi

// mpirun -np 2 --allow-run-as-root ./NCCL_hello_mpi

#include <mpi.h> 
#include <stdio.h> 
int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv); // 初始化MPI环境 
    int world_rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // 获取当前进程的Rank 
    int world_size; 
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // 获取总进程数 
    printf("Hello world from processor %d out of %d processors\n", world_rank, world_size);
    MPI_Finalize(); // 结束MPI环境 
    return 0; 
}
