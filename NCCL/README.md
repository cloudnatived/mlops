```
https://developer.nvidia.com/nccl/nccl-download

Download NCCL 2.26.2, for CUDA 12.8, March 13th, 2025
Download NCCL 2.26.2, for CUDA 12.4, March 13th, 2025   #选择所需版本
Download NCCL 2.26.2, for CUDA 12.2, March 13th, 2025

# 选择下载安装
https://developer.nvidia.com/downloads/compute/machine-learning/nccl/secure/2.26.2/ubuntu2204/x86_64/nccl-local-repo-ubuntu2204-2.26.2-cuda12.4_1.0-1_amd64.deb

# 安装
root@y248:/Data# dpkg -i nccl-local-repo-ubuntu2204-2.26.2-cuda12.4_1.0-1_amd64.deb 
Selecting previously unselected package nccl-local-repo-ubuntu2204-2.26.2-cuda12.4.
(Reading database ... 171185 files and directories currently installed.)
Preparing to unpack nccl-local-repo-ubuntu2204-2.26.2-cuda12.4_1.0-1_amd64.deb ...
Unpacking nccl-local-repo-ubuntu2204-2.26.2-cuda12.4 (1.0-1) ...
Setting up nccl-local-repo-ubuntu2204-2.26.2-cuda12.4 (1.0-1) ...

The public nccl-local-repo-ubuntu2204-2.26.2-cuda12.4 GPG key does not appear to be installed.
To install the key, run this command:
sudo cp /var/nccl-local-repo-ubuntu2204-2.26.2-cuda12.4/nccl-local-960AB412-keyring.gpg /usr/share/keyrings/

# 查看包安装情况
dpkg -l|grep nccl

# dpkg 查看软件包的安装位置
dpkg -L nccl-local-repo-ubuntu2204-2.26.2-cuda12.4

# 选择网络安装
Network Installer for Ubuntu22.04
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
$ sudo dpkg -i cuda-keyring_1.1-1_all.deb
$ sudo apt-get update

make CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local/lib/python3.10/dist-packages/nvidia/nccl -lnccl

# 选择源码安装，git，编译
https://github.com/NVIDIA/nccl/tree/v2.19
cd nccl
make -j src.build
make src.build CUDA_HOME=/usr/local/cuda
make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70“

apt install build-essential devscripts debhelper fakeroot
make pkg.debian.build

ls build/pkg/deb/
root@y248:/Data/BBC/nccl# ls -l build/pkg/deb/libnccl*
-rw-r--r-- 1 root root 115853160 Apr 16 23:20 build/pkg/deb/libnccl2_2.26.2-1+cuda12.4_amd64.deb
-rw-r--r-- 1 root root 118245864 Apr 16 23:20 build/pkg/deb/libnccl-dev_2.26.2-1+cuda12.4_amd64.deb
![图片](https://github.com/user-attachments/assets/986e6b8a-687d-4ee5-a86d-2417f1dabb8d)


NCCL-TEST
https://github.com/NVIDIA/nccl-tests

如果 CUDA 未安装在 /usr/local/cuda 中，您可以指定 CUDA_HOME。同样，如果 NCCL 未安装在 /usr 中，则可以指定 NCCL_HOME。
 make CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
NCCL 测试依赖于 MPI 在多个进程上工作，因此需要多个节点。如果要使用 MPI 支持编译测试，则需要设置 MPI=1 并将 MPI_HOME 设置为 MPI 的安装路径。
make MPI=1 MPI_HOME=/path/to/mpi CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl

root@y248:/Data/BBC/nccl-tests# ls build/*
all_gather_perf  all_reduce_perf  alltoall_perf  broadcast_perf  gather_perf  hypercube_perf  reduce_perf  reduce_scatter_perf  scatter_perf  sendrecv_perf

NCCL测试
输入如下命令测试NCCL和NCCL-test有没有安装好。
 ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
参数说明
-b 集合通信起始（最小）数据量大小
-e 集合通信结束（最大）数据量大小
-f 乘法因子（数据量按几倍增加）
-g 参与通信的GPU数量

安装openmpi，
apt-get install openmpi-bin openmpi-doc libopenmpi-dev
运行以下命令,这里对应双机4卡，注意np后面的进程数*单个节点gpu数（-g 指定）=总的gpu数量，即之前提到的等式
总的ranks数量（即CUDA设备数，也是总的gpu数量）=（进程数）*（线程数）*（每个线程的GPU数）。mpirun -np 2 -pernode \
--allow-run-as-root \
-hostfile \
-mca btl_tcp_if_include eno2  \
-x NCCL_SOCKET_IFNAME=eno2  \
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 -c 0


```
