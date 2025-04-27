# MLOps


```
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PATH=$PATH:/usr/include/mpich-3.2-x86_64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/mpich-3.2/lib
export LD_LIBRARY_PATH=/usr/lib64/mpich-3.2/lib:$LD_LIBRARY_PATH

cat >> /etc/profile << EOF
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOF

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64

make CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local/lib/python3.10/dist-packages/nvidia/nccl -lnccl

#NCCL Allreduce
nvcc nccl-reduce.cu -o nccl-reduce -lnccl

#MPI结合NCCL
nvcc nccl-reducempi.cu -o nccl-reducempi -lnccl -lmpi -L /usr/lib64/mpich-3.2/lib/ -I /usr/include/mpich-3.2-x86_64
mpiexec -n 4 ./nccl-reducempi

#分布式一维向量的softmax算子
nvcc nccl-softmax.cu -o nccl-softmax -lnccl -lmpi -L /usr/lib64/mpich-3.2/lib/ -I /usr/include/mpich-3.2-x86_64

##Cuda stream #jacobi迭代结合NCCL和MPI的多卡算法
nvcc nccl-mpi-jacobi.cu -o nccl-mpi-jacobi -lnccl -lmpi -L /usr/lib64/mpich-3.2/lib/ -I /usr/include/mpich-3.2-x86_64 mpiexec -n 2
./nccl-mpi-jacobi，表示使用nranks=2个进程。

#jacobi迭代结合NCCL的多卡算法
nvcc nccl-jacobi.cu -o nccl-jacobi -lnccl -lmpi -L /usr/lib64/mpich-3.2/lib/ -I /usr/include/mpich-3.2-x86_64

#nccl-overlay
nvcc nccl-overlay.cu -o nccl-overlay -lnccl -lmpi -L /usr/lib64/mpich-3.2/lib/ -I /usr/include/mpich-3.2-x86_64




```
