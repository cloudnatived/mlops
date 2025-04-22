```
############################################################################################################################    NCCL
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

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   #貌似找不到文档了。
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html    
nvcc nccl-example-1.cu -o nccl-example-1  -lnccl
nvcc nccl-example-2.cu -o nccl-example-2  -lnccl  -lmpi -L /usr/lib64/mpich-3.2/lib/ -I /usr/include/mpich-3.2-x86_64
nvcc nccl-example-3.cu -o nccl-example-3  -lnccl  -lmpi -L /usr/lib64/mpich-3.2/lib/ -I /usr/include/mpich-3.2-x86_64

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


############################################################################################################################    NCCL——TEST
# NCCL-TEST
https://github.com/NVIDIA/nccl-tests

如果 CUDA 未安装在 /usr/local/cuda 中，您可以指定 CUDA_HOME。同样，如果 NCCL 未安装在 /usr 中，则可以指定 NCCL_HOME。
 make CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
NCCL 测试依赖于 MPI 在多个进程上工作，因此需要多个节点。如果要使用 MPI 支持编译测试，则需要设置 MPI=1 并将 MPI_HOME 设置为 MPI 的安装路径。
make MPI=1 MPI_HOME=/path/to/mpi CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl

# 编译支持mpi的test
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi

make CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local/lib/python3.10/dist-packages/nvidia/nccl -lnccl  # /usr/local/lib/python3.10/dist-packages/nvidia/nccl 这个目录一直没找到，也没安装好这个库。


root@y248:/Data/BBC/nccl-tests# ls build/*
all_gather_perf  all_reduce_perf  alltoall_perf  broadcast_perf  gather_perf  hypercube_perf  reduce_perf  reduce_scatter_perf  scatter_perf  sendrecv_perf

# NCCL测试
输入如下命令测试NCCL和NCCL-test有没有安装好。
 ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
参数说明
-b 集合通信起始（最小）数据量大小
-e 集合通信结束（最大）数据量大小
-f 乘法因子（数据量按几倍增加）
-g 参与通信的GPU数量

# 安装openmpi
apt-get update
apt-get install infiniband-diags ibverbs-utils libibverbs-dev libfabric1 libfabric-dev libpsm2-dev -y
apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev
apt-get install librdmacm-dev libpsm2-dev


运行以下命令,这里对应双机4卡，注意np后面的进程数*单个节点gpu数（-g 指定）=总的gpu数量，即之前提到的等式
总的ranks数量（即CUDA设备数，也是总的gpu数量）=（进程数）*（线程数）*（每个线程的GPU数）。
mpirun -np 2 -pernode \
--allow-run-as-root \
-hostfile \
-mca btl_tcp_if_include eno2  \
-x NCCL_SOCKET_IFNAME=eno2  \
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 -c 0

# apt-get install mpich



############################################################################################################################
多机多卡运行nccl-tests和channel获取

1. 安装nccl
#配置网络存储库
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

#安装特定版本
sudo apt install libnccl2=2.15.1-1+cuda11.8 libnccl-dev=2.15.1-1+cuda11.8

#确认系统nccl版本
dpkg -l | grep nccl

2. 安装openmpi
#apt安装openmpi
sudo apt-get update
sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev

#验证是否安装成功
mpirun --version

3. 单机测试
#克隆该repo
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests

# 编译支持mpi的test
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi

NCCL测试可以在多个进程、多个线程和每个线程上的多个CUDA设备上运行。进程的数量由MPI进行管理，因此不作为参数传递给测试（可以通过mpirun -np n（n为进程数）来指定）。
总的ranks数量（即CUDA设备数，也是总的gpu数量）=（进程数）*（线程数）*（每个线程的GPU数）。
可以先通过nvidia-smi topo -m命令查看机器内拓扑结构，这里是双卡，两个gpu之间连接方式是PIX（Connection traversing at most a single PCIe bridge）

在 2个 GPU 上运行 （ -g 2 ），扫描范围从 8 字节到 128MB ：

./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2
这里-b表示minBytes，-e表示maxBytes，-g表示两张卡，-f表示数据量每次乘2，如开始是8B，往后依次是16，32，64字节... -g后面的gpu数量不能超过实际的数量，否则会报如下错误- invalid Device ordinal

这里执行all_reduce操作时算法带宽（algbw）和总线带宽（busbw）是一致的，并且都是随着数据量的增大而增大。关于二者的区别可见https://github.com/NVIDIA/nccl-

4. 多机测试，双机4卡nccl执行
这里使用2个节点(126,127）。 运行mpirun命令的为头节点(这里用126)，它是通过ssh远程命令来拉起其他节点（127）的业务进程的，故它需要密码访问其他host

#在126生成RSA公钥，并copy给127即可
ssh-keygen -t rsa

ssh-copy-id -i ~/.ssh/id_rsa.pub  192.168.72.127
如果ssh的端口不是22，可以在mpirun命令后添加参数-mca plm_rsh_args "-p 端口号"，除此之外，还可以在主节点上编辑以下文件

nano ~/.ssh/config
#添加以下内容
Host 192.168.72.127
    Port 2233
指定连接到特定主机时使用的端口（例如2233），并确保在执行之前检查并设置~/.ssh/config文件的权限，使其对你的用户是私有 的：

chmod 600 ~/.ssh/config
这样配置后，当你使用SSH连接到主机192.168.72.127，SSH将使用端口2233，可以减少在‘mpirun‘命令中指定端口的需要。

然后可以进行多节点测试，节点个数对应-np 后的数字，这里新建一个hostfile内容如下，每行一个ip地址就可以

192.168.72.126 192.168.72.127
mpirun -np 2 -hostfile hostfile  -pernode \
bash -c 'echo "Hello from process $OMPI_COMM_WORLD_RANK of $OMPI_COMM_WORLD_SIZE on $(hostname)"'

多节点运行nccl-tests
运行以下命令,这里对应双机4卡，注意np后面的进程数*单个节点gpu数（-g 指定）=总的gpu数量，即之前提到的等式
总的ranks数量（即CUDA设备数，也是总的gpu数量）=（进程数）*（线程数）*（每个线程的GPU数）。
mpirun -np 2 -pernode \
--allow-run-as-root \
-hostfile \
-mca btl_tcp_if_include eno2  \
-x NCCL_SOCKET_IFNAME=eno2  \
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 -c 0

避免每次命令加--allow-run-as-root
echo 'export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1' >> ~/.bashrc
echo 'export OMPI_ALLOW_RUN_AS_ROOT=1' >> ~/.bashrc

channel获取
channel的概念： nccl中channel的概念表示一个通信路径，为了更好的利用带宽和网卡，以及同一块数据可以通过多个channel并发通信，nccl会使用多channel，搜索的过程就是搜索出来一组channel。

具体一点可以参考以下文章： 如何理解Nvidia英伟达的Multi-GPU多卡通信框架NCCL？ - Connolly的回答 - 知乎 https://www.zhihu.com/question/63219175/answer/2768301153

获取channel： mpirun命令中添加参数-x NCCL_DEBUG=INFO \即可，详细信息就会输出到终端

mpirun -np 2 -pernode \
-hostfile hostfile \
-mca btl_tcp_if_include eno2 \
-x NCCL_SOCKET_IFNAME=eno2  \
-x NCCL_DEBUG=INFO  \
-x NCCL_IGNORE_DISABLED_P2P=1 \
-x CUDA_VISIBLE_DEVICES=0,1 \
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 -c 0

多机多卡运行nccl-tests和channel获取    https://zhuanlan.zhihu.com/p/682530828    Mr.King




############################################################################################################################


nvcc slzg-nccl-reduce.cu -o slzg-nccl-reduce -lnccl

nvcc slzg-nccl-reducempi.cu -o slzg-nccl-reducempi -lnccl -lmpi -L /usr/lib/x86_64-linux-gnu/openmpi/lib/ -I /usr/lib/x86_64-linux-gnu/openmpi/include/  #编译通过
执行命令为：mpiexec -n 4 ./reducempi

nvcc slzg-nccl-softmax.cu -o slzg-nccl-softmax -lnccl -lmpi -L /usr/lib/x86_64-linux-gnu/openmpi/lib/ -I /usr/lib/x86_64-linux-gnu/openmpi/include/     #编译通过

nvcc slzg-nccl-jacobi.cu -o slzg-nccl-jacobi -lnccl -lmpi
执行命令为：mpiexec -n 2 ./jslzg-nccl-jacobi，表示使用nranks=2个进程

nvcc slzg-nccl-mpi-jacobi.cu -o slzg-nccl-mpi-jacobi -lnccl -lmpi -L /usr/lib/x86_64-linux-gnu/openmpi/lib/ -I /usr/lib/x86_64-linux-gnu/openmpi/include/

nvcc slzg-nccl-overlay.cu -o slzg-nccl-overlay.cu -lnccl -lmpi

英伟达平台NCCL详细解读和代码介绍    https://zhuanlan.zhihu.com/p/686621494    森林之光
############################################################################################################################




```
