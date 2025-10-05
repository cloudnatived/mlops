


```
11. 分布式训练与集群通信，检查和测试GPU的nccl通信
分布式通信测试 (NCCL)
在进行分布式训练前，必须确保节点间GPU的通信（特别是通过10G网络）是健康和高效的。

https://blog.csdn.net/rjc_lihui/article/details/146154987    nccl-tests 调用参数 (来自deepseek)
https://www.sensecore.cn/help/docs/cloud-foundation/compute/acp/acpBestPractices/Job-nccl_test    【NGC 镜像】nccl-test 通信库检测最佳实践
https://zhuanlan.zhihu.com/p/682530828                 多机多卡运行nccl-tests和channel获取
https://cloud.tencent.com/developer/article/2361710          nccl-test 使用指引  

检查和测试GPU的nccl通信
方法1. 使用nccl-tests项目测试 NCCL 基础功能
方法2. 使用python的torch.distributed库

1. 使用nccl-tests
    在nvcr.io/nvidia/pytorch:23.10-py3等包含完整CUDA开发环境的容器中进行。
    克隆NVIDIA官方的nccl-tests项目，编译并运行性能测试脚本，如all_reduce_perf。
    观察输出的带宽（Bus B/W），评估其是否接近10G网络的理论上限。

2. 使用PyTorch Distributed测试
    编写Python脚本，利用torch.distributed库在多个进程/节点间进行张量广播、归约等操作。
    这可以更贴近实际训练场景，验证PyTorch分布式后端的通信能力。

# 使用的容器镜像是：pytorch:23.10-py3
docker run -it -d --shm-size=4G --gpus all --network host -v /Data:/Data nvcr.io/nvidia/pytorch:23.10-py3

11.1 使用nccl-tests项目测试 NCCL 基础功能
# 如果容器中已安装cuda
cat >> /etc/profile <<EOF
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
EOF

# 使用nccl-tests项目测试 NCCL 基础功能
https://github.com/NVIDIA/nccl-tests 
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make -j
./build/all_reduce_perf -b 8 -e 128M -f 2
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 -c 0
./build/broadcast_perf -b 128M -e 1G -f 2 -g 2 -c 1

# 如果出错，需要重新编译
make clean

nccl-tests 的常用参数
参数	说明
-b	起始数据大小（例如 128M 表示 128 MB）。
-e	结束数据大小（例如 1G 表示 1 GB）。
-f	数据大小的增长因子（例如 2 表示每次测试数据大小翻倍）。
-g	使用的 GPU 数量。
-c	检查结果的正确性（启用数据验证）。
-n	迭代次数（默认 100）。
-w	预热次数（默认 10）。
-o	集合操作类型（例如 all_reduce、broadcast、reduce 等）。
-d	数据类型（例如 float、double、int 等）。
-t	线程模式（0 表示单线程，1 表示多线程）。
-a	聚合模式（0 表示禁用，1 表示启用）。
-m	消息对齐（默认 0）。
-p	打印性能结果（默认启用）。
-l	指定 GPU 列表（例如 0,1,2,3 表示使用 GPU 0、1、2、3）。
-r	指定 rank 的数量（多节点测试时使用）。
-s	指定节点数量（多节点测试时使用）。

# 本次实验的目录：/Data/DEMO/CODE/NCCL/nccl-tests/
cd /Data/DEMO/CODE/NCCL/nccl-tests/
mpirun --allow-run-as-root ./build/all_reduce_perf -b 8 -e 128M -f 2  # 使用mpirun运行。
mpirun --allow-run-as-root ./build/broadcast_perf -b 128M -e 1G -f 2 -g 2 -c 1  # V100-PCIE-16GB GPU内存不足，会报错。
mpirun --allow-run-as-root ./build/broadcast_perf -b 128M -e 512M -f 2 -g 2 -c 1  # 可以完成
mpirun --allow-run-as-root -np 64 -N 8 ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1

# 先在同一节点测试是否能运行
mpirun --allow-run-as-root \
 -np 2 \
 -x NCCL_DEBUG=INFO \
 -x CUDA_VISIBLE_DEVICES=0,1 \
 ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 -c 0

# 多迭代测试​
# 增加迭代次数以获得更稳定的性能数据
mpirun --allow-run-as-root ./build/all_reduce_perf -b 1M -e 64M -f 2 -g 2 -i 100

# 减少预热迭代
mpirun --allow-run-as-root ./build/broadcast_perf -b 1M -e 64M -f 2 -g 2 -w 0 -i 50

# ​故障排除专用测试
# 最小数据量测试（排除内存问题）
mpirun --allow-run-as-root ./build/all_reduce_perf -b 1 -e 1 -f 1 -g 2

# 单字节测试
mpirun --allow-run-as-root ./build/broadcast_perf -b 1 -e 1 -f 1 -g 2

# 24.2. mpirun 选项
https://docs.redhat.com/zh-cn/documentation/red_hat_enterprise_linux/8/html/building_running_and_managing_containers/con_the-mpirun-options_assembly_using-podman-in-hpc-environment

以下 mpirun 选项用于启动容器：
--mca orte_tmpdir_base /tmp/podman-mpirun line 告诉 Open MPI 在 /tmp/podman-mpirun 中创建所有临时文件，而不是在 /tmp 中创建。如果使用多个节点，则在其他节点上这个目录的名称会不同。这需要将完整的 /tmp 目录挂载到容器中，而这更为复杂。
mpirun 命令指定要启动的命令（ podman 命令）。以下 podman 选项用于启动容器：

run 命令运行容器。
--env-host 选项将主机中的所有环境变量复制到容器中。
-v /tmp/podman-mpirun:/tmp/podman-mpirun 行告诉 Podman 挂载目录，Open MPI 在该目录中创建容器中可用的临时目录和文件。
--userns=keep-id 行确保容器内部和外部的用户 ID 映射。
--net=host --pid=host --ipc=host 行设置同样的网络、PID 和 IPC 命名空间。
mpi-ring 是容器的名称。
/home/ring 是容器中的 MPI 程序。


# 多节点运行nccl-tests，未完成。
mpirun --allow-run-as-root -np 2 -pernode \
-hostfile hostfile \
-mca btl_tcp_if_include enp4s1 \
-x NCCL_SOCKET_IFNAME=enp4s1  \
-x NCCL_DEBUG=INFO  \
-x NCCL_IGNORE_DISABLED_P2P=1 \
-x CUDA_VISIBLE_DEVICES=0,1 \
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 -c 0





11.2 使用python的torch.distributed库测试 NCCL 基础功能
# 5个python程序，测试 NCCL 基础功能
ddp_test.py
ddp_test_0.py
cuda_p2p_test.py
multi_node_nccl_test.py
advanced_nccl_test_0.py

11.2.1 ddp_test.py
python3 ddp_test.py


11.2.2 ddp_test_0.py
python3 ddp_test_0.py


11.3 cuda_p2p_test.py
python3 cuda_p2p_test.py


11.4 multi_node_nccl_test.py
# 单机测试单个GPU
python3 multi_node_nccl_test.py 0 1 172.18.8.209 29500

# 单机测试多个GPU（2个GPU）
# 一个终端运行
python3 multi_node_nccl_test.py 0 2 172.18.8.209 29500
# 另一个终端运行
python3 multi_node_nccl_test.py 1 2 172.18.8.209 29500


11.5 advanced_nccl_test_0.py
# 单机测试单个GPU
python3 advanced_nccl_test_0.py 0 1 localhost 12355

# 单机测试多个GPU（2个GPU）
# 一个终端运行
python3 advanced_nccl_test_0.py 0 2 localhost 12355
# 另一个终端运行
python3 advanced_nccl_test_0.py 1 2 localhost 12355

# 多机多GPU测试（2节点，每节点4GPU）：
节点1 (172.18.8.208)​​：
python3 advanced_nccl_test_0.py 0 6 172.18.8.208 12355
python3 advanced_nccl_test_0.py 1 6 172.18.8.208 12355

节点2 (172.18.8.209)​​：
python3 advanced_nccl_test_0.py 2 6 172.18.8.208 12355
python3 advanced_nccl_test_0.py 3 6 172.18.8.208 12355

节点3 (172.18.8.210)​​：
python3 advanced_nccl_test_0.py 4 6 172.18.8.208 12355
python3 advanced_nccl_test_0.py 5 6 172.18.8.208 12355


NCCL 调优
设置 NCCL_DEBUG、NCCL_IB_DISABLE、NCCL_P2P_DISABLE、NCCL_SOCKET_IFNAME 等参数的作用。


```





```
确保所有f-string的引号都正确配对
建议的调试方法
在运行之前，可以先用Python的语法检查工具检查：
# 使用pyflakes检查语法
pip install pyflakes
pyflakes advanced_nccl_test_0.py

# 或者使用flake8
pip install flake8
flake8 advanced_nccl_test_0.py

# 或者简单地用Python编译检查
python3 -m py_compile advanced_nccl_test_0.py

advanced_nccl_test_0.py 这是一个​​分布式NCCL性能测试工具​​，用于评估多GPU节点间的通信性能。主要功能包括：
​​测试各种集合通信操作​​：All-Reduce、All-Gather、Reduce-Scatter、Broadcast、All-to-All
​​测量通信性能​​：计算带宽和延迟
​​验证分布式环境​​：检查节点间连通性
​​生成性能报告​​：汇总测试结果

参数详解：
​​1.rank​​ (整数)
当前进程的全局排名
范围：0 到 world_size-1
示例：0, 1, 2, 3
​​
2.world_size​​ (整数)
参与测试的总进程数（通常等于总GPU数量）
示例：4（4个GPU）, 8（8个GPU）
​​
3.master_addr​​ (字符串)
主节点的IP地址或主机名
示例："192.168.1.100", "localhost", "cluster-node-01"

​​4.master_port​​ (整数)
主节点监听的端口号
示例：29500, 12345, 54321


advanced_nccl_test_0.py
# 单机多GPU测试（2个GPU）
# 一个终端运行
python3 advanced_nccl_test_0.py 0 2 localhost 12355
# 另一个终端运行
python3 advanced_nccl_test_0.py 1 2 localhost 12355

# 多机多GPU测试（2节点，每节点4GPU）：
节点1 (172.18.8.208)​​：
python3 advanced_nccl_test_0.py 0 6 172.18.8.208 12355
python3 advanced_nccl_test_0.py 1 6 172.18.8.208 12355

节点2 (172.18.8.209)​​：
python3 advanced_nccl_test_0.py 2 6 172.18.8.208 12355
python3 advanced_nccl_test_0.py 3 6 172.18.8.208 12355

节点3 (172.18.8.210)​​：
python3 advanced_nccl_test_0.py 4 6 172.18.8.208 12355
python3 advanced_nccl_test_0.py 5 6 172.18.8.208 12355


```

PyTorch安装与NCCL库版本不兼容。
```
完整解决方案​​
​​1. 验证当前NCCL版本​​
# 检查系统NCCL版本
dpkg -l | grep nccl  # Ubuntu
rpm -qa | grep nccl  # CentOS

# 或者直接查询库版本
strings /usr/lib/x86_64-linux-gnu/libnccl.so.2 | grep NCCL_
​​2. 重新安装匹配的PyTorch和NCCL​​
​​方案A：升级NCCL（推荐）​​

# Ubuntu/Debian
sudo apt install libnccl2 libnccl-dev=2.18.3-1+cuda11.8

# CentOS/RHEL 
sudo yum install nccl-2.18.3-1.cuda11.8
​​方案B：降级PyTorch​​

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
​​3. 完全清理后重新安装​​
# 彻底卸载PyTorch和NCCL
pip uninstall torch torchvision torchaudio
sudo apt purge libnccl*

# 重新安装（以CUDA 11.8为例）
sudo apt install libnccl2=2.18.3-1+cuda11.8 libnccl-dev=2.18.3-1+cuda11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
​​4. 验证修复​​
import torch
print(torch.cuda.nccl.version())  # 应该输出(2, 18, 3)

​​版本匹配原则​​：
PyTorch 2.1+ 需要 NCCL >= 2.12
CUDA 11.x 对应 NCCL 2.12-2.18
CUDA 12.x 对应 NCCL 2.18+

```



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

# 编译支持mpi的test，Tesla V100-PCIE-16GB
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi

make CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local/lib/python3.10/dist-packages/nvidia/nccl -lnccl        # /usr/local/lib/python3.10/dist-packages/nvidia/nccl 下面这个命令能找到。

pip install nvidia-nccl-cu11
https://pypi.tuna.tsinghua.edu.cn/packages/ac/9a/8b6a28b3b87d5fddab0e92cd835339eb8fbddaa71ae67518c8c1b3d05bae/nvidia_nccl_cu11-2.21.5-py3-none-manylinux2014_x86_64.whl



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

mpirun -np 16 \
  -H 172.18.8.209:2,172.18.8.210:2 \
  --allow-run-as-root -bind-to none -map-by slot \
  -x NCCL_DEBUG=INFO \
  -x NCCL_IB_GID_INDEX=3 \
  -x NCCL_IB_DISABLE=0 \
  -x NCCL_SOCKET_IFNAME=eth0 \
  -x NCCL_NET_GDR_LEVEL=2 \
  -x NCCL_IB_QPS_PER_CONNECTION=4 \
  -x NCCL_IB_TC=160 \
  -x LD_LIBRARY_PATH -x PATH \
  -mca coll_hcoll_enable 0 -mca pml ob1 -mca btl_tcp_if_include eth0 -mca btl ^openib \
  all_reduce_perf -b 32M -e 1G -i 1000 -f 2 -g 1

mpirun --allow-run-as-root -bind-to none -map-by slot all_reduce_perf_mpi -b 2048M -e 8192M -f 2 -g 1

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

# 编译支持mpi的test，Tesla V100-PCIE-16GB
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

#NCCL实现Allreduce
nvcc slzg-nccl-reduce.cu -o slzg-nccl-reduce -lnccl

#MPI结合NCCL
nvcc slzg-nccl-reducempi.cu -o slzg-nccl-reducempi -lnccl -lmpi -L /usr/lib/x86_64-linux-gnu/openmpi/lib/ -I /usr/lib/x86_64-linux-gnu/openmpi/include/  #编译通过
执行命令为：mpiexec -n 4 ./reducempi

#分布式一维向量的softmax算子实现
nvcc slzg-nccl-softmax.cu -o slzg-nccl-softmax -lnccl -lmpi -L /usr/lib/x86_64-linux-gnu/openmpi/lib/ -I /usr/lib/x86_64-linux-gnu/openmpi/include/     #编译通过

#jacobi迭代结合NCCL和MPI的多卡算法
nvcc slzg-nccl-jacobi.cu -o slzg-nccl-jacobi -lnccl -lmpi
执行命令为：mpiexec -n 2 ./jslzg-nccl-jacobi，表示使用nranks=2个进程

#jacobi迭代结合NCCL的多卡算法，不用MPI，只通过nccl来编写多卡代码
nvcc slzg-nccl-mpi-jacobi.cu -o slzg-nccl-mpi-jacobi -lnccl -lmpi -L /usr/lib/x86_64-linux-gnu/openmpi/lib/ -I /usr/lib/x86_64-linux-gnu/openmpi/include/

#计算通信重叠
nvcc slzg-nccl-overlay.cu -o slzg-nccl-overlay.cu -lnccl -lmpi

英伟达平台NCCL详细解读和代码介绍    https://zhuanlan.zhihu.com/p/686621494    森林之光
############################################################################################################################




```


参考文档：
How To Install nvidia-opencl-dev on Ubuntu 22.04    https://installati.one/install-nvidia-opencl-dev-ubuntu-22-04/
