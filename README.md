


1. 安装操作系统
2. 安装GPU驱动，V100的GPU驱动、CUDA ToolKit
3. 安装docker、containerd、nvidia-container-toolkit
4. 配置 DCGM + Prometheus + Grafana 的 GPU 监控方案
5. 开箱即用的推理服务:Ollama与Dify
6. 在单机2个 V100-PCIE-16GB 的 GPU服务器上 ，部署模型进行推理
7. 在外网访问所部署的模型推理服务
8. 分布式计算集群 Ray
9. 高性能推理引擎 vLLM 单机部署
10. 使用 vllm + ray 集群，进行多机多卡的部署测试
11. 分布式训练与集群通信，检查和测试 GPU 的 nccl 通信
11.1 使用 nccl-tests 项目测试 NCCL 基础功能
11.2 使用 python 的 torch.distributed 库测试 NCCL 基础功能
12. V100-PCIE-16GB上，ResNet-152对CIFAR-10数据集进行分布式训练
13. V100-PCIE-16GB上，使用 BERT 模型对 bert-base-uncased 数据集进行分布式训练
14. triton-inference-server部署
15. Milvus
16. Weaviate
17. NVIDIA Nsight Compute 和 NVIDIA Nsight Systems



```


1.安装操作系统
在云主机上配置V100GPU

华为的官方文档提示，在FushComputer上，安装准备用于GPU的云主机时需要注意。3点。
必备事项
前提条件
1. 虚拟机的内存全部预留。
2. 虚拟机已绑定PCI设备所在的主机。
3. 虚拟机的状态显示“已停止”。

# 因为目前这套华为私有云系统CPU和内存资源紧张，目前选择的云主机的配置:
CPU:8C
MEMORY:32G
DISK:500G

集群硬件配置概览:
服务器节点: 3台
IP地址: 172.18.8.208, 172.18.8.209, 172.18.8.210
GPU: 每台2 x V100-PCIE-16GB (算力 7.0)
CPU/内存: 8核 / 32GB
网络: 10G

GPU集群，集群的资源为，3台GPU服务器。每个GPU服务器配置为2个V100-PCIE-16GB，服务器之间的网络为10G。
集群资源配置分析:
硬件配置:
    3台服务器 × 2个V100-PCIE-16GB = 6个GPU
    每个GPU 16GB显存
    服务器间10Gb网络连接
理论计算能力:
    总显存:6 × 16GB = 96GB
    总计算单元:6 × V100
    网络带宽:10Gbps（约1.25GB/s）

V100 16GB显存限制了单卡可训练模型大小
10G网络带宽在大规模分布式训练中可能成为瓶颈
总计6块GPU的规模适中，不适合超大规模模型训练


每个云主机上有2个V100-PCIE-16GB的GPU
Tesla V100（算力 7.0）

# 用于GPU的云主机安装的操作系统。
ubuntu-22.04.5-live-server-amd64.iso

为什么要使用这个版本的操作系统？
因为，V100-PCIE-16GB，Tesla V100（算力 7.0）,对应的驱动和CUDA的版本是:cuda_11.8.0_520.61.05_linux.run
https://developer.nvidia.com/cuda-11-8-0-download-archive
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# 安装完操作系统后，配置 IP 。
ip link set enp4s1 up
ip addr add 172.18.8.209/24 dev enp4s1
ip route add default via 172.18.8.1

# 写入网卡配置文件。
cat > /etc/netplan/50-cloud-init.yaml <<EOF
# network: {config: disabled}
network:
  ethernets:
    enp4s1:
      dhcp4: false
      addresses:
        - 172.18.8.208/24
      routes:
        - to: default
          via: 172.18.8.1
      nameservers:
         addresses: [8.8.8.8, 168.95.1.1]
  version: 2
EOF

# 使用 netplan 应用网络配置。
netplan apply

# 使用清华大学的 ubuntu-22.04.5-live-server-amd64.iso 的源
cp /etc/apt/sources.list /etc/apt/sources.list.original;
cat > /etc/apt/sources.list <<EOF
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-proposed main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-proposed main restricted universe multiverse
EOF

+++++++++++++++++++++++++++++++++++   注意，这段是设置ubuntu-24.04.3的源
# 使用清华大学的 ubuntu-24.04.3 的源
cat > /etc/apt/sources.list.d/ubuntu.sources <<EOF
Types: deb
URIs: https://mirrors.tuna.tsinghua.edu.cn/ubuntu/
Suites: noble noble-updates noble-security
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF
+++++++++++++++++++++++++++++++++++

# 目前内核版本号
root@x:~# uname -a
Linux x 6.8.0-71-generic #71-Ubuntu SMP PREEMPT_DYNAMIC Tue Jul 22 16:52:38 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux

# 升级之后将变成的版本号。
root@x:~# uname -a
Linux x 6.8.0-78-generic #78-Ubuntu SMP PREEMPT_DYNAMIC Tue Aug 12 11:34:18 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux

# ubuntu-22.04.5-live-server-amd64.iso 默认操作系统版本信息。
root@x:/etc/netplan# cat /etc/*releas*
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=22.04
DISTRIB_CODENAME=jammy
DISTRIB_DESCRIPTION="Ubuntu 22.04.5 LTS"
PRETTY_NAME="Ubuntu 22.04.5 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.5 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=jammy

# 写入固定配置的 resolv 配置文件
rm -rf /etc/resolv.conf

cat > /etc/resolv.conf <<EOF
nameserver 8.8.8.8
nameserver 168.95.1.1
EOF

# 更新源
# 可能会更新内核。注意，安装完 nvidia 的 GPU 驱动之后，不要升级内核，否则需要重新安装 nvidia 的 GPU 驱动。
apt update -y;
apt list --upgradable;
apt upgrade -y;

# 运行在init 3
systemctl isolate multi-user.target;
systemctl isolate runlevel3.target;
ln -sf /lib/systemd/system/multi-user.target /etc/systemd/system/default.target;
systemctl set-default multi-user.target;

#关闭不需要的服务:
systemctl list-unit-files |awk '{ print $1,$2 }'|grep enable|egrep -v "ssh|multi|systemd-resolved|wpa_" |awk '{ print $1}'|xargs -i systemctl disable {};

#确认服务已关闭:
systemctl list-unit-files |awk '{print $1,$2}'|grep enabled;

uname -r         # 查看内核版本
lsb_release -a   # 查看发行版信息（Ubuntu/Debian/CentOS等）

# 确认安装了 gcc make
apt install -y gcc make g++ net-tools

# 重启一次。
reboot

# 禁用开源 Nouveau 驱动（重要，否则安装驱动和cuda的过程会失败）
sudo bash -c 'echo -e "blacklist nouveau\noptions nouveau modeset=0" > /etc/modprobe.d/blacklist-nouveau.conf'

# 更新内核 initramfs:（重要，否则安装驱动和cuda的过程会失败）
sudo update-initramfs -u      # Ubuntu / Debian
# 或
sudo dracut --force           # CentOS / RHEL

# 如果更新了内核就要更新内核头文件。（重要，否则安装驱动和 cuda 的过程会失败）
apt-get install -y dkms build-essential linux-headers-$(uname -r)

# 再重启一次。
reboot # 重启

# 重启后确认 Nouveau 不再被加载:
lsmod | grep nouveau   # 无输出则成功

# 查看pci设备
lspci | grep -i -E "vga|nvidia"
lspci | grep -i -E "vga|nvidia"|awk '{ print $1}'|xargs -i lspci -v -s {}

# 从启动信息里查看
dmesg |grep -i -E "vga|nvidia"

# lspci 是一个列出所有 PCI 设备的工具，包括显卡。
lspci -k | grep -A 2 -E "(VGA|3D)"

# lshw（List Hardware）可以列出系统的所有硬件配置。
lshw -C display

# 使用 ubuntu-drivers devices 查看推荐驱动，查看系统推荐的显卡驱动:
ubuntu-drivers devices

# 使用 modinfo 查看加载的驱动模块，查看当前加载的显卡驱动模块。（安装完GPU驱动后才可查看）
modinfo nvidia







2.安装GPU驱动，V100的GPU驱动、CUDA ToolKit

# Ubuntu 22.04 自带仓库已包含 525/535 等新版本
# 525+ 已适配 5.15 内核，不再使用 GPL-only 符号，无需自己编译。
sudo apt update
sudo apt install -y nvidia-driver-525   # 或 535/545

# 安装完驱动之后的提示:
root@x:/Data/IMAGES# ./cuda_11.8.0_520.61.05_linux.run 
===========
= Summary =
===========

Driver:   Not Selected
Toolkit:  Installed in /usr/local/cuda-11.8/

Please make sure that
 -   PATH includes /usr/local/cuda-11.8/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-11.8/lib64, or, add /usr/local/cuda-11.8/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.8/bin
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 520.00 is required for CUDA 11.8 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

Logfile is /var/log/cuda-installer.log


# 如果需要卸载驱动 CUDA Toolkit 11.8 
nvidia-uninstall 


# 选择run文件进行安装
# https://developer.nvidia.com/cuda-toolkit-archive

cuda_11.8.0_520.61.05_linux.run

┌─┐
│ CUDA Installer se Agreement                                                  │
│ - [ ] Driver                                                                 │
│      [ ] 520.61.05                                                           │
│ + [X] CUDA Toolkit 11.8                                                      │
│   [X] CUDA Demo Suite 11.8                                                   │
│   [X] CUDA Documentation 11.8                                                │
│ - [ ] Kernel Objects                                                         │
│      [ ] nvidia-fs                                                           │
│   Options                                                                    │
│   Install                                                                    │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│   reface                                                                     │
│                                                                              │
│                                                                              │

# 环境变量。目前添加的环境变量。之后可能需要添加 nccl ，以及其它的变量
cat >> /etc/profile <<EOF
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
EOF

source /etc/profile

# 检查 DKMS 状态
# DKMS（Dynamic Kernel Module Support）用于自动编译和安装内核模块。检查 DKMS 是否已正确配置:
dkms status

# 检查内核模块加载情况
# 重启后，再次检查内核模块是否已加载:
lsmod | grep -i nvidia
lsof | grep -i nvidia

# 测试重新加载nvidia 模块，是否报错。 
modprobe nvidia

# 查看显卡驱动版本
cat /proc/driver/nvidia/version

# 查看 CUDA、cuDNN 版本
cat /usr/local/cuda/version.json

# 查看nvcc版本
/usr/local/cuda/bin/nvcc -V

# 查看已安装的cuda-toolkit信息。
apt list |grep cuda-toolkit

NVIDIA Nsight Compute，cuda-toolkit 附带的工具。
# 查看Nsight-Compute支持的sections
ncu --list-sections

NVIDIA Nsight Systems，cuda-toolkit 附带的工具。
# 列出当前所有活动的性能分析会话。
nsys sessions list

开始使用 nvidia 系统工具
# 使用 nvidia-smi
nvidia-smi
nvidia-smi -L # 列出系统中所有可用的 GPU 设备及其 UUID。

nvidia-smi topo --matrix  # 查看 GPU 与系统其他设备的连接拓扑。

# 每隔2秒刷新一次，每次只在固定位置刷新
watch -n 5 -d nvidia-smi

# 定时查询
nvidia-smi -l 5

# 验证GPU计算能力​
# 在宿主机执行，Tesla V100（算力 7.0）
nvidia-smi --query-gpu=compute_cap --format=csv
# 输出为:
compute_cap
7.0
7.0

# 验证 GPU P2P 支持​
# 在宿主机执行
nvidia-smi topo -m

# 设备监控命令，以滚动条形式显示 GPU 设备统计信息:
nvidia-smi pmon

# 配置 GPU 功耗与算力模式：
# 查看当前模式
nvidia-smi -q -d CLOCK,POWER | grep -E "GPU Current Clock|Power Limit"

# 检查 GPU 健康状态
nvidia-smi -q -d MEMORY,UTILIZATION,POWER,CLOCK,COMPUTE

# 运行简单的 CUDA 测试
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv

# 调整功耗上限（需root，单位瓦特）
nvidia-smi -pl 250

# 锁定最高算力（针对V100的算力7.0）
nvidia-smi -ac 877,1530  # 内存频率,核心频率

# NVLink 带宽测试与优化：使用nvidia-smi nvlink --status检查链路状态，通过nccl-tests的all_reduce_perf验证跨卡通信效率。
Tesla V100-PCIE-16GB 不支持 nvlink

# GPU 资源隔离
# 使用 cgroups 限制容器 GPU 使用率：
# 创建 cgroup 限制 GPU 利用率不超过80%
# 首先设置计算模式
nvidia-smi -c 0

# 然后设置内存时钟
nvidia-smi -lmc 80

# 设置时钟频率（需要同时指定核心和内存）
nvidia-smi -ac 1590,870  # 核心1590MHz，内存870MHz

# 只设置内存时钟（如果支持）
nvidia-smi -lmc 870


# 此时还并未安装 nccl 相关工具
dpkg -l|grep nccl
hi  libnccl2                        2.15.5-1+cuda11.8                       amd64        NVIDIA Collective Communication Library (NCCL) Runtime

# 添加apt源:
# 添加 NVIDIA 官方仓库
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update

# 然后搜索所需要的软件包
apt search nvtx

# Ubuntu/Debian 
apt install -y libnccl-dev libnccl2   
# 解决:../verifiable/verifiable.cu:4:10: fatal error: nccl.h: No such file or directory
    4 | #include <nccl.h>


# 检查 NCCL 库是否存在
ls /usr/lib/x86_64-linux-gnu/libnccl*  # Ubuntu
ls /usr/local/cuda/lib64/libnccl*      # CUDA 目录
strings /usr/lib/x86_64-linux-gnu/libnccl.so | grep "NCCL" 

# 测试 NCCL 基础功能，这个测试最好使用 pytorch:23.10-py3 这类镜像来部署，否则安装库依赖很麻烦
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make
./build/all_reduce_perf -b 8 -e 128M -f 2

# 运行简单的CUDA测试
python3 -c "
import torch
print('CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU数量:', torch.cuda.device_count())
    print('当前GPU:', torch.cuda.current_device())
    print('GPU名称:', torch.cuda.get_device_name(0))
    
    # 简单的CUDA操作测试
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print('CUDA矩阵乘法测试成功')
"
CUDA可用: True
GPU数量: 2
当前GPU: 0
GPU名称: Tesla V100-PCIE-16GB
CUDA矩阵乘法测试成功


# 查看 nccl-tests 是用哪个 CUDA 编译的（如果知道路径）
ldd ./build/all_reduce_perf | grep cuda

# 由于官网的放在github上，访问很慢所以这里使用国内的存储库，中科大的源，但是中科大的源，只有 nvidia-container-toolkit 的包，并没有其它更多的安装包。（如果执行了上一步，安装了英伟达的源，就不需要再装这个源）
# 这一段不需要执行。
curl -fsSL https://mirrors.ustc.edu.cn/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://mirrors.ustc.edu.cn/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://nvidia.github.io#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://mirrors.ustc.edu.cn#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 安装nvidia-container-toolkit。
# 安装nvidia-container-toolkit，是需要在安装完docker或者containerd之后再执行的安装步骤。
apt update
apt-get install -y nvidia-container-toolkit

# 验证安装
apt list --installed *nvidia*
nvidia-container-cli  --version

# 检查 PyTorch 和 CUDA 版本匹配
# 确保 PyTorch 和 CUDA 版本匹配。你可以通过以下命令检查 PyTorch 是否能检测到 GPU:
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
"

# 安装python
apt install -y python3-pip python3 python3-netaddr wget git;
apt install -y python3-dev;
pip install --upgrade pip;

# 如何设置阿里源
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
pip config set install.trusted-host mirrors.aliyun.com

# 或者设置清华大学的
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

# 解决ubuntu24.04 使用pip时的信息提醒:error: externally-managed-environment
mv /usr/lib/python3.12/EXTERNALLY-MANAGED /usr/lib/python3.12/EXTERNALLY-MANAGED.bk

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple;
mkdir -p /root/.config/pip;
cat > /root/.config/pip/pip.conf <<EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
EOF

扩展知识:
----------------------------------------------------
批量安装驱动、CUDA、docker、NCCL 测试工具。

----------------------------------------------------





3. 安装 docker、containerd、nvidia-container-toolkit
docker、containerd、nvidia-container-toolkit，用于在GPU服务器上启动容器时，附带--gpus all这个参数。
2025-08-24，对于安装docker或者containerd的说明。目前大部分开发文档，或者一些简易工具的发布，都用的是cdocker这个命令行工具来操作。但是在一些企业级部署的时候，用的是nerdctl这个命令行工具操作。
大部分docker compose部署，都是用docker compose操作的。nerdctl工具，在操作nerdctl compose时，会缺乏参数，无法执行某些操作。
比如:dify-1.7.2版本，使用docker compose可以操作，但是用nerdctl compose，无法操作。

docker这个前端工具，操作containerd这个后台服务。

# 安装一些常用的支持。
sudo apt install apt-transport-https ca-certificates curl gnupg2 software-properties-common

# 使用清华的docker-ce的源，docker-ce的源不在常用软件包的源里:
curl -fsSL https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository -y "deb [arch=amd64] https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu $(lsb_release -cs) stable"

sudo apt install -y docker-ce docker-ce-cli containerd.io # docker的基础环境，安装这些

# 安装 nvidia-container-toolkit 。
sudo apt install -y nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container-tools libnvidia-container1

# 检查服务启动情况。
systemctl list-unit-files |grep -E "docker|contain"

root@x:/Data# systemctl list-unit-files |grep -E "docker|contain"
container-getty@.service                     static          -
containerd.service                           enabled         enabled
docker.service                               enabled         enabled
docker.socket                                enabled         enabled

# 使用 containerd 生成配置文件。这一步是必须的
containerd config default > /etc/containerd/config.toml

# 不需要配置/etc/containerd/daemon.json文件 

# 检查 runtime_type 的指向
root@x:/etc/docker# docker info |grep run
  /var/run/cdi
 Runtimes: runc io.containerd.runc.v2 nvidia
 Default Runtime: runc
 runc version: v1.2.5-0-g59923ef

# 测试运行 nvidia-container-toolkit 是否已经生效。如果正常运行，则已配置好 nvidia-container-toolkit 。
docker run --rm --gpus all nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 nvidia-smi

扩展知识:
----------------------------------------------------
K8s + GPU Operator 或 Slurm 管理多机 GPU 任务。

----------------------------------------------------





4. 配置 DCGM + Prometheus + Grafana 的 GPU 监控方案

系统架构:
云主机:172.18.8.208 2个V100-PCIE-16GB
云主机:172.18.8.209 2个V100-PCIE-16GB
云主机:172.18.8.210 2个V100-PCIE-16GB
服务器:172.18.6.64  服务器上无GPU，运行一个nvcr.io/nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04的容器，容器内运行:3个dcgm-exporter进程，从3个GPU云服务器获取信息。服务上，运行1个prometheu进程，运行1个grafana进程。

dcgm-exporter 

 # 目前 dcgm 的版本
root@x:~# dcgmi --version
dcgmi  version: 3.3.9

# 安装完成之后，dcgm.service 其实并不能顺利执行，nv-hostengine --service-account nvidia-dcgm -b ALL 这条命令。
#systemctl enable dcgm.service 
#systemctl restart dcgm.service 

root@x:/Data# systemctl list-unit-files |grep dcgm
dcgm.service                                 disabled        enabled
nvidia-dcgm.service                          disabled        enabled

# 配置 rc-local，来执行开机启动 'nv-hostengine --service-account nvidia-dcgm -b ALL'
vim /etc/systemd/system/rc-local.service  # 创建这个文件

touch /etc/systemd/system/rc-local.service
touch /etc/rc.local
chmod 775 /etc/rc.local
cat >> /etc/systemd/system/rc-local.service << EOF
[Unit]
Description=/etc/rc.local Compatibility
ConditionPathExists=/etc/rc.local
[Service]
Type=forking
ExecStart=/etc/rc.local start
TimeoutSec=0
StandardOutput=tty
RemainAfterExit=yes
#SysVStartPriority=99
[Install]
WantedBy=multi-user.target
EOF

cat >> /etc/rc.local << EOF
#!/bin/bash
nv-hostengine --service-account nvidia-dcgm -b ALL
EOF

systemctl enable rc-local.service
systemctl restart rc-local.service

# 测试dcgmi
dcgmi discovery --host 172.18.8.208 -l
dcgmi discovery --host 172.18.8.209 -l
dcgmi discovery --host 172.18.8.210 -l

# 常用命令示例
# 实时监控GPU指标（每2秒刷新）:
# -e 指定指标ID（203=GPU利用率，252=显存使用率）
# -i 指定GPU索引（0表示第一块GPU）
dcgmi discovery --host 172.18.8.210 dmon -i 0 -e 203,252 -c 5
dcgmi dmon -i 0 -e 203,252 -c 5

# 查看 GPU 健康状态:
# 检查GPU 0的健康状态（-c 表示全面检测）
dcgmi health -g 0 -c

# 统计 NVLink 带宽:
# 显示GPU 0的NVLink状态及带宽
dcgmi nvlink -i 0 -s

# 下载 dcgm 的镜像。
docker pull nvcr.io/nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04  # 容器里的dcgm为3.3.9
docker pull nvidia/dcgm-exporter:4.4.0-4.5.0-ubuntu22.04
docker pull nvcr.io/nvidia/k8s/dcgm-exporter:4.4.0-4.5.0-ubuntu22.04

# 如果使用带 GPU 的服务器来运行 dcgm-exporter 容器。
docker run -d --gpus all --cap-add SYS_ADMIN --name dcgm -p 9400:9400 -p 9401:9401 -p 9403:9403 -p 9405:9405 nvcr.io/nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04
docker run -d --gpus all --cap-add SYS_ADMIN --name dcgm -p 9400:9400 -p 9401:9401 -p 9403:9403 -p 9405:9405 nvcr.io/nvidia/k8s/dcgm-exporter:4.4.0-4.5.0-ubuntu22.04

# 找了一个 cpu 服务器来运行
docker run -d --name dcgm -p 9400:9400 -p 9401:9401 -p 9403:9403 -p 9405:9405 nvcr.io/nvidia/k8s/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04
docker run -d --name dcgm -p 9400:9400 -p 9401:9401 -p 9403:9403 -p 9405:9405 nvcr.io/nvidia/k8s/dcgm-exporter:4.4.0-4.5.0-ubuntu22.04

docker exec -it dcgm  /bin/bash     # 进入 dcgm-exporter 容器
dcgm-exporter -a :9401 -r "172.18.8.208:5555" &  # 让命令在后台持续运行。
dcgm-exporter -a :9403 -r "172.18.8.209:5555" &  # 让命令在后台持续运行。
dcgm-exporter -a :9405 -r "172.18.8.210:5555" &  # 让命令在后台持续运行。

curl 172.18.8.210:5555

# curl 测试
curl 172.18.6.64:9401
curl 172.18.6.64:9403
curl 172.18.6.64:9405

# 在172.18.6.64，下载 prometheus
https://github.com/prometheus/prometheus/releases/download/v3.6.0-rc.0/prometheus-3.6.0-rc.0.linux-amd64.tar.gz

# 修改配置，
./prometheus.yml
++++++++++++++++++++++++++++++++++++++++++++++++
    static_configs:
      - targets: ["172.18.8.208:9090"]
       # The label name is added as a label `label_name=<label_value>` to any timeseries scraped from this config.
        labels:
          app: "prometheus"

  - job_name: "DCGM_exporter"
    static_configs:
      #- targets: ["172.18.8.208:9400", "172.18.8.208:9403", "172.18.8.208:9405"]
      - targets: ["172.18.8.208:9401", "172.18.8.208:9403", "172.18.8.208:9405"]
        labels:
          app: "DCGM_exporter"
++++++++++++++++++++++++++++++++++++++++++++++++

# 运行 prometheus
./prometheus --config.file=./prometheus.yml

# 查看prometheus刚才的配置是否生效:
http://172.18.6.64:9090/targets
curl http://172.18.6.64:9090/targets

# 安装 grafana
sudo apt-get install -y adduser libfontconfig1 musl
wget https://dl.grafana.com/grafana-enterprise/release/12.1.1/grafana-enterprise_12.1.1_16903967602_linux_amd64.deb
sudo dpkg -i grafana-enterprise_12.1.1_16903967602_linux_amd64.deb

systemctl enable grafana-server.service
systemctl restart grafana-server.service

# 在 grafana 的网站查找 NVIDIA DCGM Exporter。
https://grafana.com/search/ 
12239
22515







5. 开箱即用的推理服务:Ollama与Dify
介绍如何快速部署对社区友好的模型服务，验证GPU推理能力。在任一GPU节点上，使用Docker运行Ollama容器，并映射模型存储卷和端口。
在带有GPU卡的云主机上运行ollama
# 在带有GPU卡的云主机上，启动ollama的容器
root@x:/etc/docker# docker info |grep run
  /var/run/cdi
 Runtimes: runc io.containerd.runc.v2 nvidia
 Default Runtime: runc
 runc version: v1.2.5-0-g59923ef

# 使用 ollama 作为模型推理引擎。
# 自行部署的docker
docker run -d --gpus=all --ipc=host --network host -v ollama:/root/.ollama --name ollama hub.rat.dev/ollama/ollama:0.11.6

curl http://127.0.0.1:11434/api/tags

# 进入 ollama 容器，下载模型。
nerdctl exec -it da58a9c60029 /bin/bash

ollama pull qwen3:1.7b
ollama pull qwen3:8b
ollama pull qwen3:14b
ollama pull nomic-embed-text

ollama list
root@da58a9c60029:/# ollama list
NAME                       ID              SIZE      MODIFIED
nomic-embed-text:latest    0a109f422b47    274 MB    34 hours ago
qwen3:14b                  bdbd181c33f2    9.3 GB    36 hours ago
qwen3:8b                   500a1f067a9f    5.2 GB    36 hours ago
qwen3:1.7b                 8f68893c685c    1.4 GB    37 hours ago

DIFY
dify 
# 另外找1个 CPU 机器，容器方式运行 dify
# dify 的地址:https://github.com/langgenius/dify/releases/tag/1.8.0
wget https://github.com/langgenius/dify/archive/refs/tags/1.8.0.tar.gz
tar zxf 1.8.0.tar.gz
cd dify-1.8.0;
cd docker
cp .env.example .env

nerdctl pull docker.io/langgenius/dify-sandbox:0.2.12   # 测试拉取容器镜像。
nerdctl pull hub.rat.dev/docker.io/langgenius/dify-sandbox:0.2.12  # 通过hub.rat.dev代理无法下载。
nerdctl pull m.daocloud.io/docker.io/langgenius/dify-sandbox:0.2.12  # 通过m.daocloud.io代理可以下载。

cp docker-compose.yaml docker-compose.yaml.20250823  # 备份docker-compose.yaml文件。
sed -i 's#image: #image: m.daocloud.io/docker.io/#g' docker-compose.yaml #把docker-compose.yaml所有要下载的容器镜像前，加上代理。
sed -i 's#image: #image: hub.rat.dev/docker.io/#g' docker-compose.yaml #把docker-compose.yaml所有要下载的容器镜像前，加上代理。

# 启动 didy
cp .env.example .env
docker compose up -d    # 官网命令。
nerdctl compose up -d   # 我改成了nerdctl
nerdctl compose -f docker-compose.yaml up -d # 本次部署使用的启动命令。
nerdctl compose -f docker-compose.yaml down  # 本次部署使用的停止命令。

# difi-1.18.0，外网访问地址:
http://121.40.245.182:7533/apps
wangzhen2@trustfar.cn
wangzhen2@trustfar.cn
TTyy@1234

登录之后，选在设置，选择模型供应商，添加 ollama ，然后添加模型。
模型类型:LLM
模型名称:qwen3:14b
基础url:http://172.18.8.209:11434


模型类型:Text Embedding
模型名称:nomic-embed-text
基础url:http://172.18.8.209:11434

# 增大关于文件上传的限制
vim .env
UPLOAD_FILE_SIZE_LIMIT=15
UPLOAD_FILE_BATCH_LIMIT=5

# Upload image file size limit, default 10M.
UPLOAD_IMAGE_FILE_SIZE_LIMIT=10
# Upload video file size limit, default 100M.
UPLOAD_VIDEO_FILE_SIZE_LIMIT=100
# Upload audio file size limit, default 50M.
UPLOAD_AUDIO_FILE_SIZE_LIMIT=50

WORKFLOW_FILE_UPLOAD_LIMIT=10

docker compose down
docker compose up -d







6. 在单机2个 V100-PCIE-16GB 的 GPU服务器上 ，部署模型进行推理
包含6个开源模型的部署，需要下载模型，并为每个部署创建一个 python 虚拟环境。
单个GPU节点（例如172.18.8.209）的容器内，为不同类型的AI任务部署独立的推理服务。
准备工作: 启动一个CUDA容器，并将宿主机数据目录挂载进去。为每个模型创建一个独立的Python虚拟环境以隔离依赖。
模型下载: 使用huggingface-cli并设置镜像端点HF_ENDPOINT=https://hf-mirror.com来下载模型权重。

6.1 图像分类 (ViT): 部署google/vit-base-patch16-224模型，提供Web服务。
6.2 文生图 (Stable Diffusion): 部署stable-diffusion-v1-5模型。
6.3 文字识别 (OCR): 部署microsoft/trocr-base-printed模型。
6.4 语音识别 (Whisper): 部署openai/whisper-large-v3模型。
6.5 文字转语音 (TTS): 部署coqui/XTTS-v2模型。
6.6 目标检测 (YOLO): 部署YOLOv13模型。

以下是在1个 GPU 的云主机上运行1个 nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 的容器，在容器内部署几个模型，并提供推理服务。
目前下面的3个模型，都部署在172.18.8.209这个云主机里的容器中。172.18.8.209这个云主机的配置为8C，32G memory，1000G disk。
# 运行这个命令，这样把宿主机的/Data目录共享给容器的/Data目录。
docker run -it -d --shm-size=4G --gpus all --network host --cap-add SYS_ADMIN -v /Data:/Data nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
docker run -it -d --shm-size=4G --gpus all --network host --cap-add SYS_ADMIN -v /Data:/Data nvcr.io/nvidia/pytorch:23.10-py3     # 推荐使用这个镜像


# 进入容器。
docker exec -it 3cb448814bfc9392db74341d1e4f5e27ecfac107600d76ca21c0919164fcc972 /bin/bash

# 使用清华的 apt 源，更新，安装所需的操作系统包
# 使用清华的 python 源，并安装所需的 python 库


6.1. 部署图像分类模型推理。图像分类（识别整体内容类别）,google/vit-base-patch16-224
图像分类（识别整体内容类别）,google/vit-base-patch16-224
# 这里有比较详细的介绍: 
https://zhuanlan.zhihu.com/p/713616890
基于 Vision Transformer 的图像分类模型，在 ImageNet 上预训练，支持 1000 类物体分类（如猫、狗、汽车等）。
python3 -m venv /Data/DEMO/CODE/VIT/VIT
source /Data/DEMO/CODE/VIT/VIT/bin/activate
pip download -r vit_0.py.requirements.txt --dest /Data/IMAGES/whl -v   # 把requirements.txt文件中列出的包全都先下载到/Data/IMAGES/whl目录，不安装。
pip install -r vit_0.py.requirements.txt --find-links=/Data/IMAGES/whl --no-index #从/Data/IMAGES/whl寻找安装包安装，不要連接到 PyPI。
python3 /Data/DEMO/CODE/VIT/vit_0.py
deactivate

cat > vit_0.py.requirements.txt <<EOF
gradio==5.44.1
numpy==2.3.2
pandas==2.3.2
Pillow==11.3.0
torch==2.8.0
transformers==4.56.0
opencv-python
opencv-contrib-python 
opencv-python-headless
EOF

# 程序报错，及需要安装的包
ModuleNotFoundError: No module named 'cv2'   # pip install opencv-python opencv-contrib-python opencv-python-headless
ImportError: libGL.so.1: cannot open shared object file: No such file or directory # apt install -y libgl1
ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory # apt install -y libglib2.0-0
ModuleNotFoundError: No module named 'PIL'  # pip install Pillow

curl -I https://huggingface.co/google/vit-base-patch16-224
curl -I https://hf-mirror.com/google/vit-base-patch16-224

# 基本用法-下载模型
huggingface-cli download bigscience/bloom-560m --local-dir bloom-560m
# 基本用法-下载数据集
huggingface-cli download --repo-type dataset lavita/medical-qa-shared-task-v1-toy

# 图像分类模型。google/vit-base-patch16-224
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download google/vit-base-patch16-224 --local-dir /Data/DEMO/MODEL/google/vit-base-patch16-224  # 图像分类
python3 /Data/DEMO/CODE/VIT/image_2.py
port:7860
# 外网直接访问:
http://121.40.245.182:7503/


6.2. 部署文生图模型推理。stable-diffusion-v1-5/stable-diffusion-v1-5
当前使用的stable-diffusion-v1-5/stable-diffusion-v1-5模型，是英文的，所以目前只支持英文的提示词。

7861
python3 -m venv /Data/DEMO/CODE/TXT2IMG/TXT2IMG
source /Data/DEMO/CODE/TXT2IMG/TXT2IMG/bin/activate
pip download -r text2img_server.py.requirements.txt --dest /Data/IMAGES/whl -v   # 把requirements.txt文件中列出的包全都先下载到/Data/IMAGES/whl目录，不安装。
pip install -r text2img_server.py.requirements.txt --find-links=/Data/IMAGES/whl --no-index #从/Data/IMAGES/whl寻找安装包安装，不要連接到 PyPI。
python3 /Data/DEMO/CODE/TXT2IMG/text2img_server.py
deactivate

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 --local-dir /Data/DEMO/MODEL/stable-diffusion-v1-5/stable-diffusion-v1-5  # 文生圖
python3 /Data/DEMO/CODE/TXT2IMG/text2img_server.py 7861
port:7861
# 文生图模型外网直接访问外网直接访问:
http://121.40.245.182:7504/

cat > requirements.txt.text2img_server <<EOF
gradio==5.44.1
numpy==2.3.2
pandas==2.3.2
Pillow==11.3.0
torch==2.8.0
diffusers
accelerate
transformers
EOF

# 运行以下命令检查关键依赖:
pip list | grep -E "torch|diffusers|transformers|xformers"

# 如果需要的话，强制升级所有依赖:
pip install --upgrade torch diffusers transformers accelerate



6.3. 部署OCR模型推理。OCR（文字识别）,microsoft/trocr-base-printed
OCR（文字识别）
microsoft/trocr-base-printed，目前这个模型好像有问题，无法准确识别，正在测试其它的OCR模型。
端到端 OCR 模型，支持印刷体文字识别，可直接输出图像中的文本内容。

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download microsoft/trocr-base-printed --local-dir /Data/DEMO/MODEL/microsoft/trocr-base-printed  # 印刷文本识别
python3 /Data/DEMO/CODE/OCR/ocr_0.py
port:7862
# 外网直接访问:
http://121.40.245.182:7505/

python3 -m venv /Data/DEMO/CODE/OCR/OCR
source /Data/DEMO/CODE/OCR/OCR/bin/activate
pip download -r ocr_0.py.requirements.txt --dest /Data/IMAGES/whl -v   # 把ocr_0.py.requirements.txt文件中列出的包全都先下载到/Data/IMAGES/whl目录，不安装。
pip install -r ocr_0.py.requirements.txt --find-links=/Data/IMAGES/whl --no-index #从/Data/IMAGES/whl寻找安装包安装，不要連接到 PyPI。
python3 /Data/DEMO/CODE/OCR/ocr_0.py
deactivate

cat > ocr_0.py.requirements.txt <<EOF
transformers==4.44.2
torch==2.3.1
opencv-python-headless>=4.5.0
pandas>=2.0.0
pillow>=10.0.0
gradio>=4.0.0
opencv-python
opencv-contrib-python 
opencv-python-headless
EOF

pip freeze > ocr_0.py.requirements.txt  # 保存全部的whl包名称


6.4. 部署开源语音识别模型。openai/whisper-large-v3
# 这个介绍比较详细:
https://www.cnblogs.com/liupiaos/p/18465221    python系列&deep_study系列:Whisper OpenAI开源语音识别模型
https://zhuanlan.zhihu.com/p/662906303         OpenAI Whisper 新一代语音技术(更新至v3-turbo)
  
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download openai/whisper-large-v3 --local-dir /Data/DEMO/MODEL/openai/whisper-large-v3  # 开源语音识别模型
pythone3 /Data/DEMO/CODE/WHISPER/whisper_gradio_asr_8.py
port:7863
# 外网直接访问:
http://121.40.245.182:7506/

7863
python3 -m venv /Data/DEMO/CODE/WHISPER/WHISPER
source /Data/DEMO/CODE/WHISPER/WHISPER/bin/activate
pip download -r whisper_gradio_asr_8.py.requirements.txt --dest /Data/IMAGES/whl -v   # 把requirements.txt文件中列出的包全都先下载到/Data/IMAGES/whl目录，不安装。
pip install -r whisper_gradio_asr_8.py.requirements.txt --find-links=/Data/IMAGES/whl --no-index #从/Data/IMAGES/whl寻找安装包安装，不要連接到 PyPI。
python3 /Data/DEMO/CODE/WHISPER/whisper_gradio_asr_8.py
deactivate


6.5. 部署开源语音识别模型，文字转语音。coqui/XTTS-v2
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download coqui/XTTS-v2 --local-dir /Data/DEMO/MODEL/coqui/XTTS-v2  # 开源语音识别模型，文字转语音
python3 /Data/DEMO/CODE/TTS/xtts_tts_server_5.py
port:7864
# 开源语音识别模型，文字转语音。
http://121.40.245.182:7507/

cat > requirements.txt <<EOF
gradio==5.45.0
gradio_client==1.13.0
torch==2.3.1
torchaudio==2.3.1
TTS==0.22.0
numpy
pandas
EOF

7864
python3 -m venv /Data/DEMO/CODE/TTS/TTS
source /Data/DEMO/CODE/TTS/TTS/bin/activate
pip download -r xtts_tts_server_5.py.requirements.txt --dest /Data/IMAGES/whl -v   # 把requirements.txt文件中列出的包全都先下载到/Data/IMAGES/whl目录，不安装。
pip install -r xtts_tts_server_5.py.requirements.txt --find-links=/Data/IMAGES/whl --no-index #从/Data/IMAGES/whl寻找安装包安装，不要連接到 PyPI。
python3 /Data/DEMO/CODE/TTS/xtts_tts_server_5.py
deactivate


6.6. 开源语音识别模型，文字转语音。YOLO
# 参考:
https://blog.csdn.net/youcans/article/details/142510400    【跟我学YOLO】YOLO13（2）模型下载、环境配置与检测
https://zhuanlan.zhihu.com/p/1920777603847021007

https://docs.ultralytics.com/zh/
https://github.com/iMoonLab/yolov13/
https://github.com/iMoonLab/yolov13/releases/tag/yolov13    # 下载YOLOV13模型权重文件。
port:7865
# 外网直接访问:
http://121.40.245.182:7508/


# 优先安装指定版本的Gradio（避免最新版本Schema解析问题）yolov13，一定要使用这些库。
pip install gradio==3.48.0 ultralytics==8.2.28 opencv-python==4.9.0.80 pandas==2.2.1 torch==2.2.1
pip install gradio==3.48.0 ultralytics==8.2.28 opencv-python==4.9.0.80 pandas==2.2.1 torch==2.2.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html

apt install python3.10-venv  # 需要先安装 venv
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name

7865
python3 -m venv /Data/DEMO/CODE/YOLO/YOLO
source /Data/DEMO/CODE/YOLO/YOLO/bin/activate
pip install ./flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl  --no-dependencies
pip download -r yolo_1.py.requirement.txt --dest /Data/IMAGES/whl -v   # 把requirements.txt文件中列出的包全都先下载到/Data/IMAGES/whl目录，不安装。
pip install -r yolo_1.py.requirement.txt --find-links=/Data/IMAGES/whl --no-index #从/Data/IMAGES/whl寻找安装包安装，不要連接到 PyPI。

python3 /Data/DEMO/CODE/YOLO/yolo_1.py
deactivate

# 错误提示和解决方法:
错误:ImportError: libGL.so.1: cannot open shared object file: No such file or directory
apt-get install -y libgl1-mesa-glx

错误:ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory
apt-get install -y libglib2.0-0

# 这个版本不对
(YOLO) root@anhua209:/Data/DEMO/CODE/YOLO/yolov13# pip list |grep thop
thop                     0.1.1.post2209072238

错误:ModuleNotFoundError: No module named 'thop'
pip install thop

# 运行以下命令检查关键依赖:
pip list | grep -E "torch|diffusers|transformers|xformers|ultralytics|opencv-python|pandas|onnx|flash|gradio|huggingface|thop "


(YOLO) root@anhua209:/Data/DEMO/CODE/YOLO/yolov13# cat requirements.txt 
torch==2.2.2 
torchvision==0.17.2
#flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
timm==1.0.14
albumentations==2.0.4
onnx==1.14.0
onnxruntime==1.15.1
pycocotools==2.0.7
PyYAML==6.0.1
scipy==1.13.0
onnxslim==0.1.31
onnxruntime-gpu==1.18.0
#gradio==4.44.1
gradio==3.50.2
opencv-python==4.9.0.80
psutil==5.9.8
py-cpuinfo==9.0.0
huggingface-hub==0.23.2
safetensors==0.4.3
numpy==1.26.4
supervision==0.22.0







7. 在外网访问所部署的模型推理服务

7.1. # 图像分类模型。google/vit-base-patch16-224
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download google/vit-base-patch16-224 --local-dir /Data/google/vit-base-patch16-224  # 图像分类
python3 /Data/DEMO/CODE/VIT/image_2.py
port:7860
# 外网直接访问:
http://121.40.245.182:7503/


7.2. # 文生图模型。stable-diffusion-v1-5/stable-diffusion-v1-5
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 --local-dir /Data/stable-diffusion-v1-5/stable-diffusion-v1-5  # 文生圖
python3 /Data/DEMO/CODE/TXT2IMG/text2img_server.py 7861
port:7861
# 外网直接访问:
http://121.40.245.182:7504/


7.3. # OCR模型。microsoft/trocr-base-printed
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download microsoft/trocr-base-printed --local-dir /Data/microsoft/trocr-base-printed  # 印刷文本识别
python3 /Data/DEMO/CODE/OCR/ocr_0.py
port:7862
# 外网直接访问:
http://121.40.245.182:7505/


7.4. # 开源语音识别模型。openai/whisper-large-v3
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download openai/whisper-large-v3 --local-dir /Data/DEMO/MODEL/openai/whisper-large-v3  # 开源语音识别模型
pythone3 /Data/DEMO/CODE/WHISPER/whisper_gradio_asr_8.py
port:7863
# 外网直接访问:
http://121.40.245.182:7506/


7.5. # 开源语音识别模型，文字转语音。coqui/XTTS-v2
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download coqui/XTTS-v2 --local-dir /Data/DEMO/MODEL/coqui/XTTS-v2  # 开源语音识别模型，文字转语音
python3  /Data/DEMO/CODE/TTS/xtts_tts_server_5.py
port:7864
# 开源语音识别模型，文字转语音。
http://121.40.245.182:7507/


7.6. # 开源语音识别模型，文字转语音。YOLO
https://docs.ultralytics.com/zh/
https://github.com/iMoonLab/yolov13/
https://github.com/iMoonLab/yolov13/releases/tag/yolov13  # 权重下载
python3 /Data/DEMO/CODE/YOLO/yolo_1.py
port:7865
# 外网直接访问:
http://121.40.245.182:7508/







8. 分布式计算集群Ray
RAY
docker 方式部署ray集群，未部署成功，版本太高。:
docker pull hub.rat.dev/rayproject/ray-ml:nightly-py39-gpu # 镜像里的cuda版本是:CUDA Version 12.1.1，未部署成功，版本太高。
docker pull hub.rat.dev/rayproject/ray:nightly-py39-cu128  # 镜像里的cuda版本是:CUDA Version 12.8.1，未部署成功，版本太高。
docker pull hub.rat.dev/rayproject/ray:2.37.0

# 可以下载 Ray 的 CPU 版本，在 cpu 的计算节点上做测试:
docker pull hub.rat.dev/rayproject/ray:nightly-py39-cpu # 可以下载。
docker pull m.daocloud.io/rayproject/ray:nightly-py39-cpu

# 启动 Head 节点
# 在GPU的主机上，运行一个cuda:11.8.0-cudnn8-runtime-ubuntu22.04容器，在这个容器里安装vllm，ray，并做为ray的Head节点。
# 因为需要python低于Python 3.11，ubuntu22.04的python版本是3.10.12，在CPU节点上启动一个ubuntu22.04用来安装ray。
docker run -it -d --cap-add SYS_ADMIN --net=host --pid=host --ipc=host --privileged nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 每个节点上启动一个镜像，用来部署 ray ，如果需要挂载 vllm 模型目录，每个准备作为 ray 的节点的容器，都需要预先就挂载上
docker run -it -d --gpus all --cap-add SYS_ADMIN --net=host --pid=host --ipc=host --privileged -v /Data/Qwen/Qwen2.5-1.5B:/model nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 
docker run -it -d --gpus all --cap-add SYS_ADMIN --net=host --pid=host --ipc=host --privileged -v /Data/Qwen/Qwen2.5-14B:/model nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

docker run --rm --cap-add SYS_ADMIN --net=host --pid=host --ipc=host --privileged hub.rat.dev/rayproject/ray:2.37.0


# 下载了54个 python 包
# 把安装包都下载到了/Data/IMAGES/whl/目录中，大概了解一下ray所需的python包。
-rw-r--r-- 1 root root 212986727 Sep  9 02:23 xformers-0.0.23.post1-cp310-cp310-manylinux2014_x86_64.whl

# 在 python 安装文件的目录里执行
ls |grep whl|xargs -i pip install {} --no-dependencies

# 安装 ray
pip install ray[default] --no-dependencies --find-links=/Data/IMAGES/whl
pip install "numpy<2" "transformers<4.40" "torch==2.1.2"
pip install "numpy<2" "transformers<4.40" "torch==2.1.2" --no-dependencies --find-links=/Data/IMAGES/whl

# ray 的版本
ray --version
ray, version 2.49.1

# 启动 Head 节点，在172.18.8.210节点上先启动ray，作为 Head 节点启动 ray
ray stop # 先执行stop，确保ray处在停止状态
ray stop --force
ray start --head --node-ip-address=172.18.8.208 --port=6379 --dashboard-host=0.0.0.0 --include-dashboard=true --dashboard-port=8265 --num-cpus=4 --num-gpus=2

# 启动 Worker 节点，在172.18.8.208和172.18.8.209这两个GPU云主机上启动ray，作为work节点
ray start --address='172.18.8.210:6379' --node-ip-address=172.18.8.208 --num-cpus=4 --num-gpus=2
ray start --address='172.18.8.210:6379' --node-ip-address=172.18.8.209 --num-cpus=4 --num-gpus=2

# 验证集群状态​
# Head 节点上，查看集群节点
ray list nodes
ray status
# 预期输出应包含所有节点和资源

# RAY ，测试API连通性，在任意节点上执行
curl -v http://172.18.8.210:8265/api/v0/nodes

扩展知识:
----------------------------------------------------
訓練任務是直接失敗，還是能夠（如果使用了 Ray 或其他彈性框架）檢測到節點丟失並嘗試恢復或繼續運行

在CPU服务器上使用hub.rat.dev/rayproject/ray:nightly-py39-cpu，做测试:
docker run -it -d --shm-size=4G --network host -v /Data:/Data hub.rat.dev/rayproject/ray:nightly-py39-cpu # 这个容器里的Ray版本为:ray, version 3.0.0.dev0
----------------------------------------------------


# 启动 Head 节点，
ray start --head --node-ip-address=172.18.6.60 --port=6379 --dashboard-host=0.0.0.0 --include-dashboard=true --dashboard-port=8265 --num-cpus=4

# 启动 Worker 节点，
ray start --address='172.18.6.60:6379' --node-ip-address=172.18.6.61 --num-cpus=4
ray start --address='172.18.6.60:6379' --node-ip-address=172.18.6.62 --num-cpus=4


# 当测试写 python 程序去测试 Ray 集群，并进行操作的时候，会有版本问题:
2025-10-16 16:04:07 - ERROR - 连接Ray集群失败: Version mismatch: The cluster was started with:
    Ray: 3.0.0.dev0
    Python: 3.9.23
This process on Ray Client was started with:
    Ray: 2.50.0
    Python: 3.12.3

docker pull rayproject/ray:2.50.0-cpu
docker pull hub.rat.dev/rayproject/ray:2.50.0-cpu

docker run -it -d --shm-size=4G --network host -v /Data:/Data hub.rat.dev/rayproject/ray:2.5.0-cpu
docker run -it -d --shm-size=4G --network host -v /Data:/Data hub.rat.dev/rayproject/ray:2.50.0-cpu

# 进入到任意的 Ray 节点的容器，运行:
pthone3 ray_cluster_test.py

# 提示:
🟡 关于​​任务失败警告（任务 1 和 6）​​:
🔒 ​​这是正常的，是您自己代码中故意抛出异常来测试容错性的！​

# ✅ 最终结论:
 ​​Ray 集群功能测试整体运行成功​​，具体包括:
功能                 状态                 说明
Ray 集群连接         ✅ 成功              解决了版本匹配问题后连接正常
基本任务调度 & 并发   ✅ 成功              多任务并行执行
容错能力测试          ✅ 成功              有任务故意失败，Ray 正常处理
分布式任务通信        ✅ 成功              任务间可互相通信
大数据传输测试        ✅ 成功（但有警告）    10MB 数据传输成功，但建议优化方式





9. 高性能推理引擎 vLLM 单机部署
vLLM常见问题
1. CUDA Out of Memory: 模型权重或 KV Cache 超出显存。解决方法是减小 batch size、序列长度，或使用更强的量化。
2. 模型架构不识别: Transformers库版本过低，需要升级。
3. P2P能力警告: V100 PCIE版本可能不支持P2P，添加--disable-custom-all-reduce参数可规避此问题。

https://zhuanlan.zhihu.com/p/1916898243423500022    vLLM参数详细说明
https://blog.csdn.net/baiyipiao/article/details/141930442    vllm常用参数总结
https://www.studywithgpt.com/zh-cn/tutorial/zl0s7e    使用Docker部署vLLM

确保PyTorch版本≤2.1.2，CUDA版本为11.8，V100-PCIE-16GB
transformers>=4.40.0  # Qwen3需要≥4.40

对于 ​​Tesla V100-PCIE-16GB​​，以下是经过验证的兼容CUDA容器版本:
​​1. 推荐版本矩阵​​
容器版本	                               CUDA版本	PyTorch	   TensorRT	  兼容性	 推荐度
nvcr.io/nvidia/pytorch:23.10-py3	      11.8	   2.1.2	    8.6.1	    ✅ 最佳	⭐⭐⭐⭐⭐
nvcr.io/nvidia/tensorrt:23.09-py3	      11.8	   -	        8.6.      ✅ 优秀	⭐⭐⭐⭐
nvidia/cuda:11.8.0-runtime-ubuntu20.04	11.8	   需安装	     -	       ✅ 稳定	 ⭐⭐⭐
nvcr.io/nvidia/pytorch:22.12-py3	      11.7	   1.14	      8.5.3   	✅ 良好	⭐⭐⭐

不兼容版本警告​​
​​避免以下版本​​:
容器版本	                         问题
nvcr.io/nvidia/pytorch:24.0*	    CUDA 12.4，V100支持不完善
nvidia/cuda:12.*	                需要驱动535+，可能不兼容旧系统
任何包含PyTorch                      2.2+的版本	已弃用计算能力7.0

# 硬件是 Tesla V100（算力 7.0），但日志中 PyTorch 明确提示:The minimum cuda capability supported by this library is 7.5（最低支持算力 7.5，如 Tesla T4）。
# 当前 vLLM 镜像（v0.10.1.1）内置的 PyTorch 版本过高（推测 ≥2.1），已移除对算力 7.0 的支持。
# v100建议使用 vllm/vllm-openai:v0.3.0 镜像
# vllm/vllm-openai:v0.3.0	核心:旧版镜像内置 PyTorch 2.0，支持算力 7.0 的 V100 GPU
# --disable-custom-all-reduce	禁用 vLLM 自定义的分布式通信优化（部分新版优化不兼容 V100），确保 NCCL 通信正常

# 适用 vllm/vllm-openai:v0.3.0 镜像，使用 Qwen2.5-1.5B 的模型，进行调试

适合 V100 16GB 的模型
# 推荐模型大小
MODELS=(
    "Qwen2-7B-Instruct"     # 7B参数，适合2个V100
    "Llama-2-7b-chat-hf"    # 7B参数
    "Mistral-7B-Instruct"   # 7B参数
    "Baichuan2-7B-Chat"     # 7B参数
)

# 启动参数优化
OPTIMAL_CONFIG="--tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --dtype float16 \
    --enforce-eager \
    --max-num-batched-tokens 2048"


hub.rat.dev/vllm/vllm-openai:v0.3.0                     # 适用 V100-PCIE-16GB 型号
hub.rat.dev/lmsysorg/sglang:v0.3.6.post3-cu124          # 适用 V100-PCIE-16GB 型号，但是 sglang 的这个版本已移除对算力7.0的支持。

hub.rat.dev/vllm/vllm-openai:v0.10.1.1                  # 已移除对算力7.0的支持。
m.daocloud.io/vllm/vllm-openai:v0.10.1.1                # 已移除对算力7.0的支持。
nvcr.io/nvidia/pytorch:24.05-py3
nvcr.io/nvidia/pytorch:24.02-py3
nvcr.io/nvidia/pytorch:23.10-py3
nvcr.io/nvidia/tensorrt-llm/release:0.21.0rc0
lmsysorg/sglang:v0.4.6.post4-cu124
nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3     # 这些镜像启动，需要带模型目录才能完成启动
nvcr.io/nvidia/tritonserver:25.08-pyt-python-py3
nvcr.io/nvidia/tritonserver:25.08-trtllm-python-py3
m.daocoud.io/lmsysorg/sglang:v0.5.0rc2-cu129-gb200

# 如果使用非 vllm-openai 容器镜像部署的话，需要在容器镜像内安装 vllm
pip install vllm==0.3.0
pip install "numpy<2" "transformers<4.40"
pip install "numpy<2" "transformers<4.40" "torch==2.1.2"  # 在主节点，从节点上都执行
对应的torch版本:torch-2.1.2-cp310-cp310-manylinux1_x86_64.whl
对应的vllm版本:vllm-0.3.0-cp310-cp310-manylinux1_x86_64.whl
部署vllm的时候会需要triton这个依赖，版本是:triton-2.1.0-0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl


Qwen/Qwen3-32B 
在V100-PCIE-16GB上使用ollama可以部署Qwen/Qwen3-32B进行推理，但是无法使用新版的vllm部署Qwen/Qwen3-32B。
1. ​​FP16精度部署​
模型规模	参数量	所需显存	你的配置支持情况
Qwen2-32B​​	32B	~64GB	✅ ​​最佳选择​​（4卡并行）

Qwen2-32B（FP16） - 最佳平衡​​
# 使用4张 GPU 进行张量并行
python -m torch.distributed.run --nproc_per_node=4 --nnodes=3 your_deploy_script.py --model-name Qwen/Qwen2-32B  #使用torch分布式部署Qwen2-32B的命令参考。

​​所需显存​​:~64GB
​​使用GPU​​:4张V100
​​剩余资源​​:2张GPU备用或部署其他服务

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-14B --local-dir /Data/Qwen/Qwen2.5-14B    # 下载模型

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-7B --local-dir /Data/Qwen/Qwen2.5-7B      # 下载模型

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir /Data/Qwen/Qwen2.5-1.5B    # 使用较小模型，进行多机多卡的分布式部署

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-14B --local-dir /Data/Qwen/Qwen3-14B


# 参数小的模型能执行完成，大约5分钟完成，使用非 gated 替代模型<200b>，Falcon-7BB (Apache 2.0许可)，
# 代替 meta-llama/Llama-2-7b，因为 meta-llama/Llama-2-7b 在 huggingface 需要登录，而且登录之后，仍无法下载，通过 huggingface 镜像网站也无法下载
# 创建输出目录
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download meta-llama/Llama-2-7b --local-dir /Data/meta-llama/Llama-2-7b      # 下载 huggingface 上的 meta-llama 模型文件，需要登录。
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir /Data/mistralai/mmistral-7b   # 使用非 gated 替代模型<200b>，Mistral-7B (性能优于Llama2-7B)
huggingface-cli download tiiuae/falcon-7b --local-dir /Data/tiiuae/falcon-7b                 # 使用非 gated 替代模型<200b>，Falcon-7B (Apache 2.0许可)

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download daryl149/llama-2-7b-chat-hf --local-dir /Data/meta-llama/Llama-2-7b
huggingface-cli download daryl149/llama-2-7b-chat-hf --local-dir /Data/llama-2-7b

https://www.modelscope.cn/models/LLM-Research/llama-2-7b  #  下载，llama-2-7b
git clone LLM-Research/llama-2-7b
git clone https://www.modelscope.cn/LLM-Research/llama-2-7b.git


VLLM
# 尝试，参数参考
# 使用nvcr.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04，在一个cpu节点上运行vllm服务。--network host咋个模式下，不需要要设置-p参数，参数参考
docker run -it -d --network host --cap-add SYS_ADMIN -v /Data/Qwen/Qwen3-32B:/Data/Qwen/Qwen3-32B nvcr.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04
a2e72019f4d4
pip install vllm

docker pull hub.rat.dev/vllm/vllm-openai:v0.10.1.1    # V100-PCIE-16GB上，算力不匹配，torch版本不匹配，在本次部署中无用
docker pull m.daocloud.io/vllm/vllm-openai:v0.10.1.1  # V100-PCIE-16GB上，算力不匹配，torch版本不匹配，在本次部署中无用
docker pull m.daocloud.io/vllm/vllm-openai:v0.3.0     # 算力 7.0 的支持，V100-PCIE-16GB上算力和torch版本匹配
docker pull hub.rat.dev/vllm/vllm-openai:v0.3.0       # 算力 7.0 的支持，V100-PCIE-16GB上算力和torch版本匹配

# 运行 vllm 服务的参数参考
["python3" "-m" "vllm.entrypoints.openai.api_server"]
python3 -m vllm.entrypoints.openai.api_server --model /Data/Qwen/Qwen3-32B \
  --served-model-name Qwen3_32B --host 0.0.0.0 --port 6800 --block-size 16 \
  --pipeline-parallel-size 2 --trust-remote-code --enforce-eager \
  --distributed-executor-backend ray --ray-cluster-address 172.18.6.64:6379

# hhub.rat.dev/vllm/vllm-openai:v0.3.0，在单个节点上，使用2个 V100-PCIE-16GB ，部署
# 这个是在 GPU 节点上可以启动的，在172.18.8.210上启动了，云主机配置是8C，32G，2个 V100-PCIE-16GB

# 单机双卡 V100-PCIE-16GB 部署，可以成功启动 Qwen2.5-1.5B ，成功启动:
docker run --rm \
  --gpus all \
  --net=host --pid=host --ipc=host --privileged --shm-size=4G \
  -v /Data/Qwen/Qwen2.5-1.5B:/model \
  -e VLLM_HOST_IP=172.18.8.210 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_SWAP_SPACE=1 \
  hub.rat.dev/vllm/vllm-openai:v0.3.0 \
  --model /model \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --dtype half \
  --swap-space 1 \
  --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 131072 \
  --trust-remote-code


# 单机双卡 V100-PCIE-16GB 部署，加载 Qwen2.5-14B 时，成功启动:
docker run --rm \
  --gpus all \
  --net=host --pid=host --ipc=host --privileged --shm-size=4G \
  -v /Data/Qwen/Qwen2.5-14B:/model \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e RAY_memory_usage_threshold=0.99 \
  -e RAY_memory_monitor_refresh_ms=0 \
  hub.rat.dev/vllm/vllm-openai:v0.3.0 \
  --model /model \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --dtype half \
  --swap-space 4 \
  --gpu-memory-utilization 0.995 \
  --max-model-len 400 \
  --max-num-batched-tokens 400 \
  --max-num-seqs 16 \
  --max-paddings 16 \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --enforce-eager


# 单机双卡 V100-PCIE-16GB 部署，加载 Qwen2.5-7B 时，成功启动:
docker run --rm \
  --net=host --pid=host --ipc=host --privileged --shm-size=4G \
  -v /Data/Qwen/Qwen2.5-7B:/model \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e RAY_memory_usage_threshold=0.99 \
  -e RAY_memory_monitor_refresh_ms=0 \
  hub.rat.dev/vllm/vllm-openai:v0.3.0 \
  --model /model \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --dtype half \
  --swap-space 4 \
  --gpu-memory-utilization 0.995 \
  --max-model-len 400 \
  --max-num-batched-tokens 400 \
  --max-num-seqs 16 \
  --max-paddings 16 \
  --trust-remote-code \
  --disable-custom-all-reduce


# 单机双卡 V100-PCIE-16GB 部署，加载 llama-2-7b 时，成功启动:
docker run --rm \
  --gpus all \
  --net=host --pid=host --ipc=host --privileged --shm-size=4G \
  -v /Data/llama-2-7b:/model \
  -e VLLM_HOST_IP=172.18.8.208 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_SWAP_SPACE=1 \
  hub.rat.dev/vllm/vllm-openai:v0.3.0 \
  --model /model \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --dtype half \
  --swap-space 1 \
  --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 4096







1. 服务健康状态检查
在发送具体推理请求前，可以先确认服务是否已成功启动并准备就绪。
vLLM 健康检查
curl -v http://172.18.8.210:8000/health

Triton Server 健康检查
curl -v http://172.18.8.208:8000/v2/health/ready

Ollama 模型列表检查
curl http://127.0.0.1:11434/api/tags


# 检查服务是否启动
curl -v http://172.18.8.208:8000/health

# 测试API请求
curl http://172.18.8.208:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 5000}'  # 测试长序列

# 模型列表 - 验证模型加载，查询模型信息
curl http://172.18.8.208:8000/v1/models

# 对话（Chat Completion）​，这个需要设置，chat_template.jinja，否则无法访问。
curl http://172.18.8.208:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/model",
        "messages": [
        {"role": "user", "content": "你好，介绍一下你自己"}
        ],
        "max_tokens": 200
    }'

# 生vLLM 文本生成 (Completions API)
curl http://172.18.8.208:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/model",
        "prompt": "如何学习深度学习？",
        "max_tokens": 150,
        "temperature": 0.8
    }' | jq .  # 用jq美化输出

# 使用 vLLM API访问:
curl http://172.18.8.208:8000/v1/completions -H "Content-Type: application/json" -d '{
  "model": "/model",
  "prompt": "Hello world",
  "max_tokens": 128
}'


# 问题集，及解决线索:
# 会出现很多报错。需要不断调整参数。
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.25 GiB. GPU 0 has a total capacty of 15.77 GiB of which 281.12 MiB is free. 
Including non-PyTorch memory, this process has 15.49 GiB memory in use. 
Of the allocated memory 15.04 GiB is allocated by PyTorch, and 4.72 MiB is reserved by PyTorch but unallocated. 
If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  
See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
(RayWorkerVllm pid=1756) WARNING 09-11 02:01:17 custom_all_reduce.py:44] Custom allreduce is disabled because your platform lacks GPU P2P capability. 
To slience this warning, specifydisable_custom_all_reduce=True explicitly.

# GPU blocks: 0
ValueError: No available memory for the cache blocks.
→ 权重本身已占满 2×V100-16GB，KV-cache 一块都分不到（0 blocks）。

Qwen2.5-14B 半精度权重 ≈ 28 GB，
2×16 GB = 32 GB → 只剩 < 4 GB 用于 KV-cache，
即使 gpu_memory_utilization=0.98 也 无法容纳最小 cache。

# V100-PCIE-16GB 上，无法部署 Qwen2.5-32B 模型。云主机配置8C，32G，2个 V100-PCIE-16GB。
# 错误信息。
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.31 GiB. GPU 0 has a total capacty of 15.77 GiB of which 1.76 GiB is free. 
Including non-PyTorch memory, this process has 14.00 GiB memory in use. 
Of the allocated memory 13.51 GiB is allocated by PyTorch, and 29.85 MiB is reserved by PyTorch but unallocated. 
If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
(RayWorkerVllm pid=100969) WARNING 09-10 05:13:58 custom_all_reduce.py:44] Custom allreduce is disabled because your platform lacks GPU P2P capability. To slience this warning, 
specifydisable_custom_all_reduce=True explicitly.

# 错误信息。
ValueError: The checkpoint you are trying to load has model type `qwen3` but Transformers does not recognize this architecture. 
This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.

# 错误信息。
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 36.00 MiB. GPU 0 has a total capacty of 15.77 GiB of which 22.25 MiB is free. 
Including non-PyTorch memory, this process has 15.74 GiB memory in use. 
Of the allocated memory 15.29 GiB is allocated by PyTorch, and 6.66 MiB is reserved by PyTorch but unallocated. 
If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  
See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
(RayWorkerVllm pid=100129) WARNING 09-10 05:09:56 custom_all_reduce.py:44] Custom allreduce is disabled because your platform lacks GPU P2P capability. 
To slience this warning, specifydisable_custom_all_reduce=True explicitly.



模型访问控制
在 Triton 或 vLLM 中集成 API 密钥认证，例如 vllm 启动时添加：
vllm serve --model qwen3:14b --api-key my-secret-key




10. 使用 vllm + ray 集群，进行多机多卡的部署测试
使用vllm+ray集群进行多机多卡的部署测试。3台GPU服务器。每个GPU服务器配置为2个V100-PCIE-16GB，服务器之间的网络为10G。
# Qwen2.5-1.5B使用4卡，不能使用6卡，因为不能整除。大约5分钟完成加载。
# Qwen2.5-7B使用4卡，未测试。
# Qwen2.5-14B使用4卡，无法完成部署，大概加载模型30分钟后。会出错:ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization` when initializing the engine.
# V100-PCIE-16GB上，无法部署Qwen2.5-14B模型。

# 在命令行里启动 vllm 的命令，启动命令参考:
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
RAY_ADDRESS=172.18.8.210:6379 \
python3 -m vllm.entrypoints.openai.api_server \
  --model /model \
  --tensor-parallel-size 4 \
  --worker-use-ray \
  --host 0.0.0.0 \
  --dtype half \
  --swap-space 1 \
  --enforce-eager \
  --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 131072 \
  --trust-remote-code \
  --disable-custom-all-reduce


# 在172.18.8.208这个节点上启动，并设置172.18.8.208容器里的 ray ，为 head 节点:
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
RAY_ADDRESS=172.18.8.208:6379 \
docker run --rm \
  --gpus all \
  --net=host --pid=host --ipc=host --privileged --shm-size=4G \
  -v /Data/llama-2-7b:/model \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_SWAP_SPACE=1 \
  -e RAY_NODE_TYPE=head \
  -e RAY_HEAD_SERVICE_HOST=172.18.8.208 \
  -e RAY_HEAD_SERVICE_PORT=6379 \
  -e RAY_DASHBOARD_PORT=8265 \
  -e VLLM_TENSOR_PARALLEL_SIZE=2 \
  hub.rat.dev/vllm/vllm-openai:v0.6.3 \
  --model /model \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --dtype half \
  --swap-space 1 \
  --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 4096 \
  --disable-custom-all-reduce \
  --worker-use-ray \
  --enforce-eager


# 在172.18.8.209节点上启动，并设置172.18.8.209容器里的 ray ，为 work 节点:
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
RAY_ADDRESS=172.18.8.208:6379 \
docker run --rm \
  --gpus all \
  --net=host --pid=host --ipc=host --privileged --shm-size=4G \
  -v /Data/llama-2-7b:/model \
  -e VLLM_HOST_IP=172.18.8.209 \
  -e RAY_HEAD_SERVICE_PORT=6379 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_SWAP_SPACE=1 \
  -e RAY_NODE_TYPE=worker \
  -e RAY_HEAD_SERVICE_HOST=172.18.8.208 \
  -e RAY_HEAD_SERVICE_PORT=6379 \
  -e RAY_DASHBOARD_PORT=8265 \
  -e VLLM_TENSOR_PARALLEL_SIZE=2 \
  -e RAY_ADDRESS=172.18.8.208:6379 \
  hub.rat.dev/vllm/vllm-openai:v0.6.3 \
  --model /model \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --dtype half \
  --swap-space 1 \
  --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 4096 \
  --disable-custom-all-reduce \
  --worker-use-ray \
  --enforce-eager


# 在172.18.8.210节点上启动，并设置172.18.8.210容器里的 ray ，为 work 节点:
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
RAY_ADDRESS=172.18.8.208:6379 \
docker run --rm \
  --gpus all \
  --net=host --pid=host --ipc=host --privileged --shm-size=4G \
  -v /Data/llama-2-7b:/model \
  -e VLLM_HOST_IP=172.18.8.209 \
  -e RAY_HEAD_SERVICE_PORT=6379 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_SWAP_SPACE=1 \
  -e RAY_NODE_TYPE=worker \
  -e RAY_HEAD_SERVICE_HOST=172.18.8.208 \
  -e RAY_HEAD_SERVICE_PORT=6379 \
  -e RAY_DASHBOARD_PORT=8265 \
  -e VLLM_TENSOR_PARALLEL_SIZE=2 \
  -e RAY_ADDRESS=172.18.8.208:6379 \
  hub.rat.dev/vllm/vllm-openai:v0.6.3 \
  --model /model \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --dtype half \
  --swap-space 1 \
  --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 4096 \
  --disable-custom-all-reduce \
  --worker-use-ray \
  --enforce-eager




# 命令行总结
  --max-num-batched-tokens 131072 \ # 不需要添加
  --gpu-memory-utilization 0.8 \ 
  -e RAY_ADDRESS="172.18.8.208:10001" \
  -e RAY_REDIS_PASSWORD="your_password" \  # RAY密码

vocab_size = 151936 无法被 6 整除，而你现在要求 --tensor-parallel-size 6。
vLLM 的 VocabParallelEmbedding 必须整除，否则会直接断言失败。

# 不支持的参数
--distributed-executor-backend ray       # 参数在vllm-openai:v0.3.0中不支持。
--enable-chunked-prefill False           # 参数在vllm-openai:v0.3.0中不支持。
--ray-cluster-address 172.18.6.64:6379   # 参数在vllm-openai:v0.3.0中不支持







11. 分布式训练与集群通信，检查和测试 GPU 的 nccl 通信
分布式通信测试 (NCCL)
在进行分布式训练前，必须确保节点间GPU的通信（特别是通过10G网络）是健康和高效的。

https://blog.csdn.net/rjc_lihui/article/details/146154987    nccl-tests 调用参数 (来自deepseek)
https://www.sensecore.cn/help/docs/cloud-foundation/compute/acp/acpBestPractices/Job-nccl_test    【NGC 镜像】nccl-test 通信库检测最佳实践
https://zhuanlan.zhihu.com/p/682530828                 多机多卡运行nccl-tests和channel获取
https://cloud.tencent.com/developer/article/2361710          nccl-test 使用指引  

检查和测试GPU的nccl通信
方法1. 使用nccl-tests项目测试 NCCL 基础功能
方法2. 使用python的torch.distributed库

1. 使用 nccl-tests
  在nvcr.io/nvidia/pytorch:23.10-py3等包含完整CUDA开发环境的容器中进行。
  克隆NVIDIA官方的nccl-tests项目，编译并运行性能测试脚本，如all_reduce_perf。
  观察输出的带宽（Bus B/W），评估其是否接近10G网络的理论上限。

2. 使用 PyTorch Distributed 测试
  编写Python脚本，利用torch.distributed库在多个进程/节点间进行张量广播、归约等操作。
  这可以更贴近实际训练场景，验证PyTorch分布式后端的通信能力。

# 使用的容器镜像是:pytorch:23.10-py3
docker run -it -d --shm-size=4G --gpus all --network host -v /Data:/Data nvcr.io/nvidia/pytorch:23.10-py3
docker run -it -d --shm-size=4G --gpus all --net=bridge -p 2222:22 -v /Data:/Data nvcr.io/nvidia/pytorch:23.10-py3   # 需要使用网桥方式。设置:--net=bridge

apt-get install -y openssh-server
cat >> /etc/ssh/sshd_config <<EOF
PermitRootLogin yes
EOF
/etc/init.d/ssh restart
passwd 


pip install -r requirements.txt

# 安装 ansible.posix 集合
ansible-galaxy collection install ansible.posix

# 安装其他Kubespray可能需要的集合
ansible-galaxy collection install community.general
ansible-galaxy collection install kubernetes.core

# 安装 ansible.utils 集合
ansible-galaxy collection install ansible.utils

# 同时安装其他可能缺少的集合
ansible-galaxy collection install ansible.posix community.general kubernetes.core

# 在主节点生成密钥
ssh-keygen -t rsa

# 复制公钥到所有节点（包括自己）
ssh-copy-id -o StrictHostKeyChecking=no root@172.18.8.208
ssh-copy-id -o StrictHostKeyChecking=no root@172.18.8.209
ssh-copy-id -o StrictHostKeyChecking=no root@172.18.8.210 

在主节点测试 hostfile 中所有节点的连通性:
# 遍历 hostfile 中的节点 IP，测试 ping 和 SSH 连接
for ip in 172.18.8.208 172.18.8.209 172.18.8.210; do
  echo "Testing $ip..."
  ping -c 2 $ip  # 测试网络连通性
  ssh root@$ip "echo $ip is reachable"  # 测试 SSH 连通性
done


# 使用的容器镜像是:pytorch:23.10-py3，并直接进入容器:
docker run -it --shm-size=4G --gpus all --network host -v /Data:/Data nvcr.io/nvidia/pytorch:23.10-py3 /bin/bash







11.1 使用 nccl-tests 项目测试 NCCL 基础功能
# 如果容器中已安装cuda
cat >> /etc/profile <<EOF
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
EOF

# 使用 nccl-tests 项目测试 NCCL 基础功能
https://github.com/NVIDIA/nccl-tests 
git clone https://github.com/NVIDIA/nccl-tests.git
cd /Data/DEMO/CODE/NCCL/nccl-tests
make -j
./build/all_reduce_perf -b 8 -e 128M -f 2
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 -c 0
./build/broadcast_perf -b 128M -e 1G -f 2 -g 2 -c 1

# 如果出错，需要重新编译
make clean

nccl-tests 项目中的常用参数
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

# 本次实验的目录:/Data/DEMO/CODE/NCCL/nccl-tests/
cd /Data/DEMO/CODE/NCCL/nccl-tests/
mpirun --allow-run-as-root ./build/all_reduce_perf -b 8 -e 128M -f 2  # 使用mpirun运行。
mpirun --allow-run-as-root ./build/broadcast_perf -b 128M -e 1G -f 2 -g 2 -c 1  # V100-PCIE-16GB GPU内存不足，会报错。
mpirun --allow-run-as-root ./build/broadcast_perf -b 128M -e 512M -f 2 -g 2 -c 1  # 可以完成
mpirun --allow-run-as-root -np 64 -N 8 ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1 # 未通过，会报错。
mpirun --allow-run-as-root -np 3 -N 3 ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1 # 未通过，会报错。

# 先在同一节点测试是否能运行
mpirun --allow-run-as-root \
 -np 2 \
 -x NCCL_DEBUG=INFO \
 -x CUDA_VISIBLE_DEVICES=0,1 \
 ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 -c 0

# 多迭代测试​
# 增加迭代次数以获得更稳定的性能数据
mpirun --allow-run-as-root -np 1 ./build/all_reduce_perf -b 1M -e 64M -f 2 -g 2 -i 100
mpirun --allow-run-as-root -np 2 ./build/all_reduce_perf -b 1M -e 64M -f 2 -g 2 -i 100
mpirun --allow-run-as-root ./build/all_reduce_perf -b 1M -e 4M -f 2 -g 2 -i 10

# 减少预热迭代
mpirun --allow-run-as-root ./build/broadcast_perf -b 1M -e 64M -f 2 -g 2 -w 0 -i 50

# ​故障排除专用测试
# 最小数据量测试（排除内存问题）
mpirun --allow-run-as-root ./build/all_reduce_perf -b 1 -e 1 -f 1 -g 2

# 单字节测试
mpirun --allow-run-as-root -np 1 ./build/broadcast_perf -b 1 -e 1 -f 1 -g 2
mpirun --allow-run-as-root -np 1 ./build/broadcast_perf -b 1M -e 64M -f 2 -g 2 -w 0 -i 50

mpirun --allow-run-as-root --mca plm_base_verbose 10 -H "192.168.1.11" -np 2 true

mpirun --allow-run-as-root --mca plm_base_verbose 10 -H "192.168.1.12" -np 2 printenv

mpirun --allow-run-as-root --mca plm_base_verbose 10 -H "192.168.1.11,192.168.1.12" -np 2 printenv


# 24.2. mpirun 选项
https://docs.redhat.com/zh-cn/documentation/red_hat_enterprise_linux/8/html/building_running_and_managing_containers/con_the-mpirun-options_assembly_using-podman-in-hpc-environment

以下 mpirun 选项用于启动容器:
--mca orte_tmpdir_base /tmp/podman-mpirun line 告诉 Open MPI 在 /tmp/podman-mpirun 中创建所有临时文件，而不是在 /tmp 中创建。
如果使用多个节点，则在其他节点上这个目录的名称会不同。这需要将完整的 /tmp 目录挂载到容器中，而这更为复杂。

mpirun 命令指定要启动的命令（ podman 命令）。以下 podman 选项用于启动容器:
run 命令运行容器。
--env-host 选项将主机中的所有环境变量复制到容器中。
-v /tmp/podman-mpirun:/tmp/podman-mpirun 行告诉 Podman 挂载目录，Open MPI 在该目录中创建容器中可用的临时目录和文件。
--userns=keep-id 行确保容器内部和外部的用户 ID 映射。
--net=host --pid=host --ipc=host 行设置同样的网络、PID 和 IPC 命名空间。
mpi-ring 是容器的名称。
/home/ring 是容器中的 MPI 程序。


# 多节点启动
# 使用mpirun多节点运行nccl-tests，因为端口通讯和主机名的问题未完成。
1. 指定基础端口
mpirun --allow-run-as-root \
  -np 6 \
  -hostfile hostfile \
  -mca btl_tcp_if_include eth0 \
  -x NCCL_SOCKET_IFNAME=eth0 \
  -x NCCL_DEBUG=INFO \
  -x NCCL_PORT=15000 \
  -x CUDA_VISIBLE_DEVICES=0,1 \
  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 -c 0

mpirun --allow-run-as-root \
  -np 6 \
  -hostfile hostfile \
  -mca btl_tcp_if_include enp4s1 \
  -x NCCL_SOCKET_IFNAME=enp4s1 \
  -x NCCL_DEBUG=INFO \
  -x NCCL_PORT=15000 \
  -x CUDA_VISIBLE_DEVICES=0,1 \
  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 -c 0

2. 指定端口范围
mpirun --allow-run-as-root \
  -np 6 \
  -hostfile hostfile \
  -mca btl_tcp_if_include eth0 \
  -x NCCL_SOCKET_IFNAME=eth0 \
  -x NCCL_DEBUG=INFO \
  -x NCCL_PORT=15000 \              # 基础端口
  -x NCCL_MIN_PORT=15000 \          # 最小端口
  -x NCCL_MAX_PORT=15100 \          # 最大端口
  -x CUDA_VISIBLE_DEVICES=0,1 \
  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 -c 0

3. 同时指定多个相关端口
mpirun --allow-run-as-root \
  -np 6 \
  -hostfile hostfile \
  -mca btl_tcp_if_include eth0 \
  -x NCCL_SOCKET_IFNAME=eth0 \
  -x NCCL_DEBUG=INFO \
  -x NCCL_PORT=15000 \
  -x NCCL_SOCKET_SEND_RECV_PREFIX=tcp \
  -x CUDA_VISIBLE_DEVICES=0,1 \
  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 -c 0

mpirun --allow-run-as-root \
  -np 6 \
  --hostfile hostfile \
  -mca btl_tcp_if_include eth0 \
  -mca btl_tcp_port_min_v4 30000 \
  -mca btl_tcp_port_range_v4 100 \
  -x NCCL_SOCKET_IFNAME=eth0 \
  -x NCCL_DEBUG=INFO \
  -x NCCL_IGNORE_DISABLED_P2P=1 \
  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1







11.2 使用 python 的 torch.distributed 库测试 NCCL 基础功能
# 如果是docker容器运行，需要在 docker 容器启动时，使用--network host
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


11.2.3 cuda_p2p_test.py
python3 cuda_p2p_test.py


11.2.4 multi_node_nccl_test.py
# 单机测试单个 GPU
python3 multi_node_nccl_test.py 0 1 172.18.8.209 29500

# 单机测试多个 GPU（2个GPU）
# 一个终端运行
python3 multi_node_nccl_test.py 0 2 172.18.8.209 29500
# 另一个终端运行
python3 multi_node_nccl_test.py 1 2 172.18.8.209 29500


11.2.5 advanced_nccl_test_0.py
# 单机测试单个 GPU
python3 advanced_nccl_test_0.py 0 1 localhost 12355

# 单机测试多个 GPU（2个GPU）
# 一个终端运行
python3 advanced_nccl_test_0.py 0 2 localhost 12355
# 另一个终端运行
python3 advanced_nccl_test_0.py 1 2 localhost 12355

# 多节点启动
# 多机多 GPU 测试（3节点，每节点2GPU）:
节点1 (172.18.8.208)​​:
python3 advanced_nccl_test_0.py 0 6 172.18.8.208 12355
python3 advanced_nccl_test_0.py 1 6 172.18.8.208 12355

节点2 (172.18.8.209)​​:
python3 advanced_nccl_test_0.py 2 6 172.18.8.208 12355
python3 advanced_nccl_test_0.py 3 6 172.18.8.208 12355

节点3 (172.18.8.210)​​:
python3 advanced_nccl_test_0.py 4 6 172.18.8.208 12355
python3 advanced_nccl_test_0.py 5 6 172.18.8.208 12355


NCCL 调优
设置 NCCL_DEBUG、NCCL_IB_DISABLE、NCCL_P2P_DISABLE、NCCL_SOCKET_IFNAME 等参数的作用。







11.3. 使用openmpi的编译器，编译和运行nccl官网上的3个示例
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
https://scc.ustc.edu.cn/zlsc/user_doc/html/mpi-application/mpi-application.html    MPI并行程序编译及运行    
https://xflops.sjtu.edu.cn/hpc-start-guide/parallel-computing/mpi/    HPC入门指南


# 安装 openmpi
apt-get update
apt-get install -y infiniband-diags ibverbs-utils libibverbs-dev libfabric1 libfabric-dev libpsm2-dev
apt-get install -y openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev
apt-get install -y librdmacm-dev # 不需要安装

在pytorch:23.10-py3容器镜像内，在/opt/hpcx/ompi/下已经安装好了openmpi。不需要要执行以上步骤。

# openmpi 的一些工具:
mpirun:
mpirun是openmpi的命令行工具，它提供了一种简单的方式来并行启动应用程序，但是必须依赖openmpi环境。
它允许在多个节点上同时启动多个并行应用程序，每个应用程序都是以进程的方式运行，而不是线程。另外，mpirun和mpiexec是同一个工具，用法相同。

mpicc

oshmem_info 是 OpenSHMEM 库提供的一个工具，用于显示 SHMEM (Scalable Hierarchical Memory) 编程环境的信息。


# 可能出现的错误提示:
mpirun: symbol lookup error: mpirun: undefined symbol: opal_libevent2022_event_base_loop    # 

测试编译个运行hello_mpi.c:
// hello_mpi.c
// check env in terminal: mpicc --version; mpirun --version
// 检查库:ldconfig -p | grep libmpi
// apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev
// gcc hello_mpi.c -o hello_mpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib -Wl,-rpath,/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lm

// 强烈推荐使用方案3（mpicc），因为它是MPI标准提供的编译器包装器，会自动处理所有必要的包含路径和库链接。
// apt-get install openmpi-bin libopenmpi-dev
// mpicc hello_mpi.c -o hello_mpi

// mpirun -np 2 --allow-run-as-root ./hello_mpi


https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html

1. Example 1: Single Process, Single Thread, Multiple Devices
nccl-official-example-1.cu 
# 修改代码中的 nDev = 1，devs[0] = {0}
nvcc nccl-official-example-1.cu -o nccl-official-example-1 -lnccl

2. Example 2: One Device per Process or Thread
这是一个结合了MPI和NCCL的分布式程序:
nccl-official-example-2.cu
nvcc nccl-official-example-2.cu -o nccl-official-example-2 -lnccl -lmpi -L /usr/lib/x86_64-linux-gnu/openmpi/lib -I /usr/lib/x86_64-linux-gnu/openmpi/include   # 出错:fatal error: mpi.h: No such file or directory
nvcc nccl-official-example-2.cpp -o nccl-official-example-2 -lnccl -lmpi -L /usr/lib/x86_64-linux-gnu/openmpi/lib -I /usr/lib/x86_64-linux-gnu/openmpi/include  # 出错:fatal error: mpi.h: No such file or directory

# 方法1:同时链接 MPI 和 NCCL ✅ 
mpicxx -o nccl-official-example-2 nccl-official-example-2.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnccl -lcudart # 可执行成功

# 方法2:如果使用系统默认路径
mpicxx -o nccl-official-example-2 nccl-official-example-2.cpp -lnccl -lcudart # 出错，fatal error: cuda_runtime.h: No such file or directory

# 方法3:指定 CUDA toolkit 路径
mpicxx -o nccl-official-example-2 nccl-official-example-2.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnccl -lcudart -L/usr/lib/x86_64-linux-gnu/openmpi/lib -I/usr/lib/x86_64-linux-gnu/openmpi/include # 可执行成功


3. Example 3: Multiple Devices per Thread
nvcc nccl-official-example-3.cu -o nccl-official-example-3  -lnccl  -lmpi -L /usr/lib64/mpich-3.2/lib/ -I /usr/include/mpich-3.2-x86_64 # 出错，fatal error: mpi.h: No such file or directory

mpicxx -o 22.out 22.c -lnccl  -lmpi -L /usr/lib64/mpich-3.2/lib/ -I /usr/include/mpich-3.2-x86_64 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart
mpicxx nccl-official-example-2.cu -o nccl-official-example-2 -lnccl  -lmpi -L /usr/lib64/mpich-3.2/lib/ -I /usr/include/mpich-3.2-x86_64 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart


mpicxx nccl-official-example-2.cpp -o nccl-official-example-2 \
  -lnccl -lmpi \
  -L/usr/lib64/mpich-3.2/lib -I/usr/include/mpich-3.2-x86_64 \
  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart

mpicxx nccl-official-example-3.cpp -o nccl-official-example-3 \
  -lnccl -lmpi \
  -L/usr/lib64/mpich-3.2/lib -I/usr/include/mpich-3.2-x86_64 \
  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart

mpicxx nccl-official-example-3.cpp -o nccl-official-example-3 \
  -I/usr/local/cuda/include \
  -I/usr/local/nccl/include \
  -L/usr/local/cuda/lib64 -lcuda -lcudart \
  -L/usr/local/nccl/lib -lnccl \
  -pthread

nvcc nccl-official-example-3.cpp -o nccl-official-example-3 \
  -I/usr/include/openmpi-x86_64 \
  -L/usr/lib64/openmpi/lib \
  -lmpi -lnccl -lcudart -O2




12. V100-PCIE-16GB上，ResNet-152 对 CIFAR-10 数据集进行分布式训练
参考文档:
https://blog.csdn.net/qq_41185868/article/details/82793025   Dataset之CIFAR-10:CIFAR-10数据集的简介、下载、使用方法之详细攻略
https://www.cnblogs.com/mengtao-wang/p/18888373    Pytorch实战-CIFAR-10图像分类
https://zhuanlan.zhihu.com/p/72679537     深度学习之16——残差网络(ResNet)
https://zhuanlan.zhihu.com/p/353235794?s_r=0      ResNet50网络结构图及结构详解
https://blog.csdn.net/Chuck0415/article/details/146167848     ResNet50深度解析:原理、结构与PyTorch实现

残差网络(ResNet)
残差网络在设计之初，主要是服务于卷积神经网络(CNN)，在计算机视觉领域应用较多，但是随着CNN结构的发展，在很多文本处理，文本分类里面(n-gram)，
也同样展现出来很好的效果。

分布式训练实战:ResNet-152，CIFAR-10
完整的端到端示例，在3台服务器（共6个GPU）上对CIFAR-10数据集进行长时间的分布式训练。

模型开发与优化
模型精度与速度优化
混合精度训练（AMP）
Gradient checkpointing
ZeRO / FSDP（分布式内存优化）

在3台GPU服务器上，每个GPU服务器配置为2个V100-PCIE-16GB，服务器之间的网络为10G。帮我设计一个需要24小时才能完成的分布式模型训练。给出所有具体的程序。
下面是一个具体的分布式模型训练程序示例，使用PyTorch作为深度学习框架，并使用分布式训练。这个程序将帮助您在3台GPU服务器上进行分布式训练。
CIFAR-10是一个更接近普适物体的彩色图像数据集。CIFAR-10 是由Hinton 的学生Alex Krizhevsky 和Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含10 个类别的RGB 
彩色图片:飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。
每个图片的尺寸为32 × 32 ，每个类别有6000个图像，数据集中一共有50000 张训练图片和10000 张测试图片。


# 使用的容器镜像是:pytorch:23.10-py3
docker run -it -d --shm-size=4G --gpus all --network host -v /Data:/Data nvcr.io/nvidia/pytorch:23.10-py3

# 程序文件:
cifar10_utils.py​
cifar10_train.py​
run.sh


使用run.sh来运行，内容如下:
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#!/bin/bash
# run.sh

MASTER_ADDR="172.18.8.208" # 在MASTER_ADDR的rank0上创建目录和日志，只在rank0进程上打印汇总结果，每个epoch结束时保存检查点（只在rank0上执行）， # 训练结束后保存最终模型（只在rank0上执行）
MASTER_PORT="29500"
WORLD_SIZE=6
GPU_PER_NODE=2
RANK=$1

# 可选:设置 NCCL 参数优化 10G 网络    
export NCCL_SOCKET_IFNAME=enp4s1
export NCCL_DEBUG=INFO

cd /Data/DEMO/CODE/RESNET/;
python3 cifar10_train.py \
  --rank $RANK \
  --world_size $WORLD_SIZE \
  --master_addr $MASTER_ADDR \
  --epochs 100 \
  --batch_size 32 \
  --accumulation_steps 4 \
  --data_dir ./data \
  --save_dir ./checkpoints \
  --log_dir ./logs
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

多节点启动:
在172.18.8.208:
bash /Data/DEMO/CODE/RESNET/run.sh 0 &
bash /Data/DEMO/CODE/RESNET/run.sh 1 &

在172.18.8.209:
bash /Data/DEMO/CODE/RESNET/run.sh 2 &
bash /Data/DEMO/CODE/RESNET/run.sh 3 &

在172.18.8.210:
bash /Data/DEMO/CODE/RESNET/run.sh 4 &
bash /Data/DEMO/CODE/RESNET/run.sh 5 &

# 每隔2秒刷新一次，每次只在固定位置刷新 ✅ 
watch -n 5 -d nvidia-smi

https://www.cnblogs.com/mengtao-wang/p/18888373    Pytorch实战-CIFAR-10图像分类
训练策略与超参数参考:
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

损失函数:交叉熵损失是多分类任务的标准选择，可衡量模型输出的概率分布与真实标签分布之间的差异；
优化器:SGD具有良好的稳定性和可解释性；
学习率调度:每 20 个 epoch 衰减 10 倍；
动量:momentum=0.9 能加速收敛并抑制震荡。


扩展知识:
----------------------------------------------------
哪些模型训练框架是专门用于训练大语言模型的。
以下是专门用于训练大语言模型的主要框架:
1. 主流专用框架
Megatron-LM (NVIDIA)  # NVIDIA 开发的高效 Transformer训练框架
DeepSpeed (Microsoft) # Microsoft 的深度优化训练库
FairScale (Meta)      # Meta 的可扩展训练库
2. 综合AI框架中的LLM支持
Hugging Face Transformers + Accelerate  # Hugging Face 生态系统的训练工具
PyTorch Lightning                       # 简化 PyTorch 训练的高级框架
3. 专业训练平台
Colossal-AI   # 中国的开源大模型训练框架
Alpa          # Google和UC Berkeley 合作的自动并行训练系统
4. 云平台专用解决方案
AWS SageMaker Model Parallelism    # AWS的模型并行训练服务
Google Vertex AI                   # Google Cloud的AI训练平台
5. 新兴框架
FastMoE           # 专注于MoE（Mixture of Experts）的框架
Petals            # 去中心化的LLM训练和推理
6. 框架选择建议
初学者推荐:
# Hugging Face + Accelerate
# 优点:文档丰富，社区活跃，易上手
pip install transformers accelerate datasets
企业级生产环境:
# DeepSpeed 或 Megatron-LM
# 优点:性能优化极致，支持超大规模模型
# 缺点:学习曲线陡峭
研究用途:
# FairScale + PyTorch Lightning
# 优点:灵活性高，实验方便
# 缺点:需要更多手动调优
----------------------------------------------------

扩展知识:
----------------------------------------------------
哪些常用的模型训练框架是用于训练大语言模型之外的模型的。
这些框架覆盖了机器学习的主要应用领域:
视觉任务: Detectron2, MMDetection, YOLO
语音任务: ESPnet, SpeechBrain, Kaldi
推荐系统: DeepCTR, RecBole, TFRS
图学习: PyG, DGL, GraphVite
强化学习: SB3, RLlib, Acme
时间序列: GluonTS, TSFresh
多模态: OpenMMLab系列, MONAI
AutoML: AutoGluon, H2O.ai
选择框架时需要考虑任务类型、团队技术栈、性能要求等因素。
----------------------------------------------------






13. V100-PCIE-16GB上，使用 BERT 模型对 bert-base-uncased 数据集进行分布式训练
核心目标
任务类型:BERT 模型的预训练（Pretraining）。
支持的任务:
掩码语言建模 (MLM):预测被遮蔽的单词。
在自然语言处理（NLP）领域，BERT（Bidirectional Encoder Representations from Transformers）模型已经成为一个重要的里程碑。BERT-base-uncased模型是BERT系列中的一个基础版本，适用于英文文本处理




13.1 使用 BERT 模型对 bert-base-uncased 数据集进行分布式训练
# 使用的容器镜像是:pytorch:23.10-py3
docker run -it -d --shm-size=4G --gpus all --ipc=host --network host -v /Data:/Data nvcr.io/nvidia/pytorch:23.10-py3

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple;
mkdir -p /root/.config/pip;
cat > /root/.config/pip/pip.conf <<EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
EOF

# 本程序依赖的版本
pip list | grep -E 'torch|transformers'
pip install transformers==4.30.0 huggingface_hub safetensors==0.6.2 --no-dependencies --find-links=/Data/IMAGES/whl   # 只需要安装transformers等这几个。

# 安装完成之后:
pip list | grep -E 'torch|transformers'
pytorch-quantization      2.1.2
torch                     2.1.0a0+32f93b1
torch-tensorrt            0.0.0
torchdata                 0.7.0a0
torchtext                 0.16.0a0
torchvision               0.16.0a0
transformers              4.30.0

# 保存python包的版本和依赖:
apt install pipreqs
pipreqs ./ --savepath ../bert-requirements.txt

cat ./bert-requirements.txt # 会发现torch版本有差异，但是可以忽略
pytorch-quantization==2.1.2
torch==2.3.1
torch-tensorrt==0.0.0
torchdata==0.7.0a0
torchtext==0.16.0a0
torchvision==0.16.0a0
transformers==4.30.0


# 下載模型文件到指定目錄，每个主机都要有一份
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download bert-base-uncased --local-dir /Data/DEMO/MODEL/bert-base-uncased

# 如果出现:
⚠️  Warning: 'huggingface-cli download' is deprecated. Use 'hf download' instead.

# 下载模型文件
wget https://hf-mirror.com/bert-base-uncased/resolve/main/pytorch_model.bin -P /Data/DEMO/MODEL/bert-base-uncased
wget https://hf-mirror.com/bert-base-uncased/resolve/main/config.json -P /Data/DEMO/MODEL/bert-base-uncased
wget https://hf-mirror.com/bert-base-uncased/resolve/main/vocab.txt -P /Data/DEMO/MODEL/bert-base-uncased

# bert_train.py中，从本地加载
tokenizer = BertTokenizer.from_pretrained("./models/bert-base-uncased")
model = BertModel.from_pretrained("./models/bert-base-uncased")

scp -r * 172.18.8.209:/Data/DEMO/CODE/BERT/
scp -r * 172.18.8.210:/Data/DEMO/CODE/BERT/

# 参考文档
https://pytorch.ac.cn/docs/2.5/elastic/run.html#google_vignette    # torchrun 的文档
torchrun 是一个 python 控制台脚本，指向 torch.distributed.run 的主模块，该模块在 setup.py 中的 entry_points 配置中声明。它等效于调用 python -m torch.distributed.run。
https://blog.csdn.net/u013172930/article/details/148519788    # 【PyTorch】torchrun:分布式训练的启动和管理命令行工具
https://blog.csdn.net/Komach/article/details/130765773        #  关于集群分布式torchrun命令踩坑记录（自用）


# 模型文件和程序:
bert_dataset.py
bert_model.py
bert_train.py


# 主节点上测试运行:172.18.8.208
export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_id=100 \
    --rdzv_backend=static \
    --master_addr=172.18.8.208 \
    --master_port=29500 \
    bert_train.py --config config.yaml


# 多节点启动 (在主节点运行) 172.18.8.208
torchrun \
  --nnodes=3 \
  --nproc_per_node=2 \
  --node_rank=0 \
  --master_addr=172.18.8.208 \
  --master_port=29500 \
  bert_train.py \
  --config config.yaml

# 多节点启动，172.18.8.209
torchrun \
  --nnodes=3 \
  --nproc_per_node=2 \
  --node_rank=1 \
  --master_addr=172.18.8.208 \
  --master_port=29500 \
  bert_train.py \
  --config config.yaml

# 多节点启动，172.18.8.210
torchrun \
  --nnodes=3 \
  --nproc_per_node=2 \
  --node_rank=2 \
  --master_addr=172.18.8.208 \
  --master_port=29500 \
  bert_train.py \
  --config config.yaml




13.2 使用 BERT 模型对 bert-base-uncased 数据集进行分布式训练，添加 TensorBoard
使用 TensorBoard 可视化你的 BERT 分布式训练过程，需通过 TensorBoard 日志写入、分布式环境适配（仅主节点写日志）、关键指标追踪（损失、学习率、GPU 内存等）三个核心步骤实现。
确保 tensorboard 和 torch.utils.tensorboard 已安装:
pip install tensorboard  # TensorBoard 核心库
# PyTorch 2.1.0 已内置 torch.utils.tensorboard，无需额外安装,✅ 

# 模型文件和程序:
bert_dataset.py
bert_model.py
bert_train.py

（1）导入 TensorBoard 相关模块
在文件顶部导入 SummaryWriter（TensorBoard 日志写入工具）:
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
import torch.nn as nn
# ... 其他原有导入 ...
from torch.utils.tensorboard import SummaryWriter  # 新增:导入 TensorBoard 写入器
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

（2）初始化 TensorBoard 日志器（仅主节点
在 main() 函数中，主节点（rank 0）创建 SummaryWriter（指定日志保存路径），从节点不创建（避免冲突）。修改位置在「自动创建保存目录」之后:
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main():
    # ... 原有代码:解析参数、加载 config、初始化分布式、创建保存目录 ...

    # 新增:初始化 TensorBoard 日志器（仅主节点）
    writer = None  # 初始化 writer 为 None
    if rank == 0:
        # 日志保存路径:./tensorboard_logs/[当前时间]（避免多次训练日志混淆）
        tensorboard_log_dir = os.path.join("./tensorboard_logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_log_dir)  # 创建日志写入器
        logger.info(f"TensorBoard 日志已保存至:{tensorboard_log_dir}")

    # ... 后续代码:初始化模型、优化器、数据加载 ...
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

（3）训练过程中写入关键指标
在 train_epoch() 函数中，每次迭代 / 每 N 步 将关键指标写入 TensorBoard。需修改 train_epoch() 函数，添加 writer 和 global_step（全局步数，用于 TensorBoard 横坐标）参数:
步骤 3.1:修改 train_epoch() 函数定义
增加 writer 和 global_step 参数:
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def train_epoch(
    model, dataloader, optimizer, scheduler, criterion, scaler, 
    epoch, rank, world_size, logger, accumulation_steps,
    writer=None, global_step=None  # 新增:TensorBoard 相关参数
):
    model.train()
    total_loss = 0.0
    total_steps = len(dataloader)
    # ... 原有代码:初始化进度条、优化器清零 ...
步骤 3.2:迭代中写入指标（每 50 步，与进度条更新同步）
在 train_epoch() 函数的迭代循环中，添加指标写入逻辑（仅主节点执行）:
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def train_epoch(...):
    # ... 原有代码:迭代遍历 dataloader ...

    for step, batch in enumerate(pbar):
        # ... 原有代码:数据迁移到 GPU、前向计算、损失计算 ...

        # 原有代码:进度条更新（rank 0）
        if rank == 0 and step % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            mem_used = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GPU 内存（GB）
            pbar.set_postfix({
                'mlm_loss': f"{mlm_loss.item():.4f}",
                'nsp_loss': f"{nsp_loss.item():.4f}" if nsp_loss != 0 else "N/A",
                'lr': f"{current_lr:.2e}",
                'mem_used': f"{mem_used:.2f}GB"
            })

            # 新增:写入 TensorBoard 指标（仅主节点，每 50 步）
            if writer is not None and global_step is not None:
                writer.add_scalar("Train/MLM_Loss", mlm_loss.item(), global_step)  # MLM 损失
                writer.add_scalar("Train/NSP_Loss", nsp_loss.item(), global_step)  # NSP 损失
                writer.add_scalar("Train/Total_Loss", (mlm_loss + nsp_loss).item(), global_step)  # 总损失
                writer.add_scalar("Train/Learning_Rate", current_lr, global_step)  # 学习率
                writer.add_scalar("Train/GPU_Memory_GB", mem_used, global_step)  # GPU 内存使用
                global_step += 1  # 全局步数递增

    # ... 原有代码:计算平均损失、分布式损失同步 ...

    # 新增:每个 Epoch 结束后，写入平均损失（可选，补充 Epoch 级指标）
    if rank == 0 and writer is not None:
        writer.add_scalar("Train/Average_Loss_Per_Epoch", avg_loss, epoch)
    
    return avg_loss, global_step  # 新增:返回更新后的 global_step


（4）在 main() 中调用 train_epoch() 并传递参数
在 main() 函数的训练循环中，初始化 global_step（全局步数，初始为 0），并将 writer 和 global_step 传递给 train_epoch():
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main():
    # ... 原有代码:初始化模型、优化器、数据加载 ...

    # 新增:初始化全局步数（用于 TensorBoard 横坐标，每 50 步递增）
    global_step = 0
    best_loss = float('inf')

    # 训练循环
    for epoch in range(config['training']['epochs']):
        sampler.set_epoch(epoch)
        
        # 修改:调用 train_epoch() 时传递 writer 和 global_step，并接收更新后的 global_step
        avg_loss, global_step = train_epoch(
            model, dataloader, optimizer, scheduler, criterion, scaler,
            epoch, rank, world_size, logger, config['training']['accumulation_steps'],
            writer=writer, global_step=global_step  # 传递 TensorBoard 参数
        )

        # ... 原有代码:保存模型、日志打印 ...

    # 新增:训练结束后关闭 TensorBoard 写入器（释放资源）
    if rank == 0 and writer is not None:
        writer.close()
        logger.info("TensorBoard 日志器已关闭")

    # ... 原有代码:清理分布式环境 ...
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# 多节点启动 (在主节点运行) 172.18.8.208
torchrun \
  --nnodes=3 \
  --nproc_per_node=2 \
  --node_rank=0 \
  --master_addr=172.18.8.208 \
  --master_port=29500 \
  bert_train_tensorboard.py \
  --config config.yaml

# 多节点启动，172.18.8.209
torchrun \
  --nnodes=3 \
  --nproc_per_node=2 \
  --node_rank=1 \
  --master_addr=172.18.8.208 \
  --master_port=29500 \
  bert_train_tensorboard.py \
  --config config.yaml

# 多节点启动，172.18.8.210
torchrun \
  --nnodes=3 \
  --nproc_per_node=2 \
  --node_rank=2 \
  --master_addr=172.18.8.208 \
  --master_port=29500 \
  bert_train_tensorboard.py \
  --config config.yaml


启动 TensorBoard 服务
# 格式:tensorboard --logdir=日志保存路径 --port=端口号（避免端口冲突）
export TENSORBOARD_HOST=0.0.0.0
tensorboard --logdir=./tensorboard_logs --port=6006

# 大约7小时完成模型训练。
anhua208:6888:44995 [0] NCCL INFO comm 0x555e22e0ddc0 rank 1 nranks 6 cudaDev 1 busId d0 - Abort COMPLETE
[2025-09-30 22:04:05] __main__ INFO: [Rank 0] 保存最佳模型 - 损失: 10.8564
[2025-09-30 22:04:05] __main__ INFO: [Rank 0] 训练完成！
[2025-09-30 22:04:05] __main__ INFO: [Rank 0] 最佳损失: 10.8564
[2025-09-30 22:04:05] __main__ INFO: [Rank 0] TensorBoard 日志器已关闭
anhua208:6887:6917 [0] NCCL INFO [Service thread] Connection closed by localRank 0
anhua208:6887:45022 [0] NCCL INFO comm 0x560b56bc7700 rank 0 nranks 6 cudaDev 0 busId c0 - Abort COMPLETE



扩展知识:
----------------------------------------------------
除了TensorBoard，还有很多其他方法可以对模型训练过程进行可视化和监控:
1. Weights & Biases (W&B)
2. MLflow
3. Comet.ml
4. Neptune.ai
5. 自定义Matplotlib实时绘图
6. Plotly + Dash 实时监控面板
7. Visdom (Facebook)
8. 自定义日志文件 + 外部工具
9. 使用Pandas + Jupyter进行交互式分析
10. Prometheus + Grafana (生产环境)
11. Sacred + Omniboard
12. 简单的进度条 + 指标显示

选择建议:
研究/实验阶段: W&B, TensorBoard, MLflow
生产环境: Prometheus + Grafana, 自定义监控面板
快速原型: Matplotlib实时绘图, tqdm进度条
团队协作: W&B, MLflow, Neptune
资源受限: 简单的日志记录 + 外部分析
----------------------------------------------------






14. triton-inference-server 部署
https://blog.csdn.net/u013171226/article/details/148792425    Triton server的部署、构建、backend插件机制整体介绍
https://zhuanlan.zhihu.com/p/634444666                        模型推理服务化框架Triton保姆式教程（三）:开发实践
https://www.jianshu.com/p/e4ff723b101a                        AI模型部署:一文搞定Triton Inference Server的常用基础配置和功能特性
https://www.jianshu.com/p/b59860ce0fe9                        saved_model 转 tensorrt 的 plan 模型

扩展知识:
----------------------------------------------------
NVIDIA Triton Inference Server是为大规模部署设计的高性能推理服务器
V100适配的Triton容器版本为nvcr.io/nvidia/tritonserver:23.09-py3，其内置TensorRT 8.6.1。
TensorRT 加速，在 V100 上用 TensorRT 对 Stable Diffusion、Whisper、LLaMA 等模型做推理加速。
启动Triton SDK容器nvcr.io/nvidia/tritonserver:23.09-py3-sdk用于发送客户端请求。

GPU型号为:V100-PCIE-16GB
则算力为:7.0
则适配的CUDA版本为:cuda_11.8.0_520.61.05_linux.run    # 这个安装包里带GPU驱动。
则TensorRT版本为:TensorRT version: 8.6.1。
trtexec版本为:TensorRT v8601
tritonserver版本为:server_version:2.39.0

# 让大语言模型提问部署给出部署方案和程序:
在3台 GPU 服务器，每个 GPU 服务器配置为2个 V100-PCIE-16GB ，如何使用 tensorrt 和 triton 对 yolov13x.pt 模型进行分布式部署。
----------------------------------------------------

14.1. 环境准备
需要容器镜像为:nvcr.io/nvidia/tritonserver:23.09-py3
docker run -it -d --shm-size=4G --name tritonserver --gpus all --network host -v /Data:/Data nvcr.io/nvidia/tritonserver:23.09-py3
能在容器中直接运行:tritonserver

# triton-inference-server 在 github 上的官网。
https://github.com/triton-inference-server/server
Release 2.38.0 corresponding to NGC container 23.09  # 说明triton-inference-server对应的容器镜像版本，tritonserver:23.09-py3
https://github.com/triton-inference-server/server/archive/refs/tags/v2.38.0.tar.gz  # 下载链接

https://www.cnblogs.com/zzk0/p/15932542.html    # 我不会用 Triton 系列:命令行参数简要介绍
https://zhuanlan.zhihu.com/p/574146311          # 深度学习部署神器——triton inference server入门教程指北
https://zhuanlan.zhihu.com/p/21172600328        # 使用 triton 部署模型
https://blog.csdn.net/qq_41664845/article/details/125529197    # Triton部署Torch和Onnx模型，集成数据预处理

# 这里是官方的步骤:
++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
# Step 1: Create the example model repository
git clone -b r25.08 https://github.com/triton-inference-server/server.git  # 官方教程，本次部署不使用这个版本。
cd server/docs/examples
./fetch_models.sh

server-2.38.0 # 实际用的是这个版本:
需要下载2个文件:
https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz # 这个能直接下。
mv /tmp/inception_v3_2016_08_28_frozen.pb model_repository/inception_graphdef/1/model.graphdef
https://contentmamluswest001.blob.core.windows.net/content/14b2744cf8d6418c87ffddc3f3127242/9502630827244d60a1214f250e3bbca7/08aed7327d694b8dbaee2c97b8d0fcba/densenet121-1.2.onnx # 这个下载不了。
得到:model_repository/inception_graphdef/1/model.graphdef

https://github.com/onnx/models # 替代方法，从这里找 densenet121 对应的版本。
https://github.com/onnx/models/tree/main/validated/vision/classification/densenet-121   # 下载地址
https://github.com/onnx/models/blob/main/validated/vision/classification/densenet-121/model/densenet-7.onnx
cp densenet-7.onnx densenet121-1.2.onnx;
得到:model_repository/densenet_onnx/1/model.onnx
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

++++++++++++++++++++++++++++++++++++++++++++++++++++++++   这里是官网文档说明:
# Step 2: Launch triton from the NGC Triton container
docker run --gpus=1 --rm --net=host -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:25.08-py3 tritonserver --model-repository=/models --model-control-mode explicit --load-model densenet_onnx

# Step 3: Sending an Inference Request
# In a separate console, launch the image_client example from the NGC Triton SDK container
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.09-py3-sdk /workspace/install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg

# 测试访问服务
# 运行命令:
/workspace/install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION -u http://172.18.8.208:8000 /workspace/images/mug.jpg

# Inference should return the following
Image '/workspace/images/mug.jpg':
    15.346230 (504) = COFFEE MUG
    13.224326 (968) = CUP
    10.422965 (505) = COFFEEPOT
++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 这里是官网文档说明


# 这里是本次试验的实际步骤:
++++++++++++++++++++++++++++++++++++++++++++++++++++++++    这里是本次试验的实际步骤:
cd /Data/DEMO/CODE/server-2.38.0/docs/examples;
# 在172.18.8.208的tritonserver:23.09-py3容器内可以运行，依赖/opt/tritonserver下的很多库。
tritonserver --model-repository=./model_repository --model-control-mode explicit --load-model densenet_onnx


curl -v 172.18.8.208:8000/v2/health/ready # 验证tritonserver已正常运行
curl -v 172.18.8.209:8000/v2/health/ready

# 检查模型元数据
curl http://172.18.8.208:8000/v2/models/densenet_onnx

# 用原始 HTTP 请求:未测通。
curl -X POST http://172.18.8.208:8000/v2/models/densenet_onnx/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "input_0",
      "shape": [1, 3, 224, 224],
      "datatype": "FP32",
      "data": [1.jpg]
    }]
  }'


# 使用: nvcr.io/nvidia/tritonserver:23.09-py3-sdk 测试访问:
# 在GPU服务器上运行，对应的容器版本:
docker run -it -d --shm-size=4G --name tritonserver-sdk --gpus all --network host -v /Data:/Data nvcr.io/nvidia/tritonserver:23.09-py3-sdk

# 进入容器，运行命令:
/workspace/install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION -u http://172.18.8.208:8000 /workspace/images/mug.jpg

# 可以找一个 CPU 服务器运行:
docker run -it -d --shm-size=4G --name tritonserver-sdk --network host -v /Data:/Data nvcr.io/nvidia/tritonserver:23.09-py3-sdk

# 验证服务是否启动成功（查看日志是否有"Successfully loaded model"）
curl http://172.18.8.210:8000/v2/models/sd_pipeline                # 若返回模型信息，说明服务正常
curl -v http://172.18.8.210:8000/v2/health/ready                   # 若返回模型信息，说明服务正常

curl http://172.18.8.208:8000/v2/models/yolov13                    # 若返回模型信息，说明服务正常
curl -v http://172.18.8.208:8000/v2/health/ready                   # 若返回模型信息，说明服务正常

# 下载英伟达官方镜像
https://catalog.ngc.nvidia.com/



triton-inference-server 部署 yolov13x
进行模型转换，
pytorch 格式的模型，转换成 onnx 格式，再转换成

python3 touch2onnx.py
# 提示:ModuleNotFoundError: No module named 'ultralytics' 
# 解决方法:
pip install ultralytics

# 提示:AttributeError: module 'cv2.dnn' has no attribute 'DictValue'
# 解决方法:
pip install opencv-python opencv-contrib-python opencv-python-headless

pip install onnxruntime==1.15.1 ultralytics==8.3.213
pip install opencv-python==4.9.0.80 opencv-contrib-python==4.9.0.80 opencv-python-headless==4.9.0.80
pip install opencv-python==4.12.0.88 opencv-contrib-python==4.12.0.88 opencv-python-headless==4.12.0.88
pip install opencv==4.8.1 opencv-python==4.8.1.78 opencv-contrib-python==4.8.1.78 opencv-python-headless==4.8.1.78

# 提示:ImportError: numpy.core.multiarray failed to import
pip install numpy==1.26.4

pip list |grep -E 'transformers|ultralytics|numpy|opencv|polars|polars_runtime|onnxruntime|ultralytics'
numpy                     1.26.4
opencv                    4.7.0
opencv-contrib-python     4.12.0.88
opencv-python             4.12.0.88
opencv-python-headless    4.12.0.88
polars                    1.34.0
polars-runtime-32         1.34.0
transformers              4.30.0
ultralytics               8.3.213
ultralytics-thop          2.0.17

pip install opencv-python==4.10.0.84
pip install opencv-python-headless







15. Milvus
1. Milvus
Milvus Milvus 是一个企业级向量数据库，支持大规模数据集和高并发场景。它采用分布式架构，支持多种索引（如 HNSW 和 IVF），并提供毫秒级延迟。适合需要高性能和扩展性的应用。
https://hub.docker.com/r/milvusdb/milvus/   # hub.docker.com上的地址
docker pull milvusdb/milvus:v2.6.3-gpu  # 这个是GPU版的，3.04 GB
docker pull hub.rat.dev/milvusdb/milvus:v2.6.3      # 这个是CPU版的，1.02 GB
docker pull hub.rat.dev/milvusdb/milvus:v2.5.15      # 这个是CPU版的，1.02 GB   

Milvus 官网说明地址:
https://milvus.io/docs/zh/install_standalone-docker.md

Milvus 进行容器部署时，部署在GPU上有什么好处？ # 在大语言模型里提问。

# 参考资料:
https://blog.csdn.net/lsb2002/article/details/132222947    为AI而生的数据库:Milvus详解及实战
https://zhuanlan.zhihu.com/p/634255317   Milvus 完整指南:开源向量数据库，AI 应用开发的基础设施（逐行解释代码，小白适用）
https://www.milvus-io.com/getstarted/standalone/install_standalone-helm   用 Kubernetes 安装独立运行的 Milvus


wget https://github.com/milvus-io/milvus/releases/download/v2.5.15/milvus-standalone-docker-compose.yml -O docker-compose.yml        # 本次测试使用这个版本
wget https://github.com/milvus-io/milvus/releases/download/v2.6.0-rc1/milvus-standalone-docker-compose-gpu.yml -O docker-compose.yml # 这个是目前最新版本

# 需要使用 docker-compose
apt install -y docker-compose
docker-compose up -d
docker-compose ps

# 设置代理下载:
export http_proxy=http://192.168.1.2:7890;
export https_proxy=https://192.168.1.2:7890;

# docker 镜像

docker pull minio/minio:RELEASE.2023-03-20T20-16-18Z
docker pull milvusdb/milvus:v2.2.11

docker pull dhub.kubesre.xyz/milvusdb/milvus:v2.2.11
docker pull dhub.kubesre.xyz/minio/minio:RELEASE.2023-03-20T20-16-18Z

docker tag dhub.kubesre.xyz/milvusdb/milvus:v2.2.11 milvusdb/milvus:v2.2.11
docker tag dhub.kubesre.xyz/minio/minio:RELEASE.2023-03-20T20-16-18Z minio/minio:RELEASE.2023-03-20T20-16-18Z

WARN[0000] /Data/BBC/Milvus/docker-compose.yml: `version` is obsolete
docker ps 
CONTAINER ID   IMAGE                                                  COMMAND                  CREATED             STATUS                       PORTS                                                                                          NAMES
69deaf344a2a   hub.rat.dev/milvusdb/milvus:v2.5.15                    "/tini -- milvus run…"   About an hour ago   Up About an hour (healthy)   0.0.0.0:9091->9091/tcp, [::]:9091->9091/tcp, 0.0.0.0:19530->19530/tcp, [::]:19530->19530/tcp   milvus-standalone
217102be5b70   hub.rat.dev/minio/minio:RELEASE.2024-05-28T17-19-04Z   "/usr/bin/docker-ent…"   About an hour ago   Up About an hour (healthy)   0.0.0.0:9000-9001->9000-9001/tcp, [::]:9000-9001->9000-9001/tcp                                milvus-minio
e5634dc58a3a   quay.io/coreos/etcd:v3.5.18                            "etcd -advertise-cli…"   About an hour ago   Up About an hour (healthy)   2379-2380/tcp                                                                                  milvus-etcd

# 测试 Milvus 数据库的连接和读写:
python3 hello_milvus.py

# Python 导入 Milvus 向量数据:
pip install pymilvus sentence_transformers




扩展知识:
----------------------------------------------------
使用 Milvus Operator 在 Kubernetes 中运行 Milvus:
https://milvus.io/docs/zh/install_cluster-milvusoperator.md?tab=helm

----------------------------------------------------



2. 使用 attu 访问 Milvus
# 使用attu访问Milvus
docker pull docker.chenby.cn/zilliz/attu
docker pull dhub.kubesre.xyz/zilliz/attu:v2.2.6

docker tag dhub.kubesre.xyz/zilliz/attu:v2.2.6 zilliz/attu:v2.2.6
docker pull dhub.kubesre.xyz/zilliz/attu:v2.4.4


# 运行 Attu 容器的参考命令:
docker run -d \
  --name attu \
  -p 18000:3000 \
  -e MILVUS_URL=host.docker.internal:19530 \
  --add-host=host.docker.internal:host-gateway \
  zilliz/attu:v2.3.8

# 如果无法直接下载容器镜像，也可以通过添加 daemon.json 文件里的 registry-mirrors ，来解决直接下载的问题:
mkdir -p /etc/docker
tee /etc/docker/daemon.json <<EOF
{
    "registry-mirrors": [
        "https://docker.anyhub.us.kg",
        "https://dockerhub.icu",
        "https://docker.awsl9527.cn"
    ]
}
EOF
systemctl daemon-reload
systemctl restart docker

http://192.168.1.103:8000/
zilliz:zilliz

扩展知识:
----------------------------------------------------
Milvus 与 Weaviate 详细比较报告    https://www.syndataworks.cn/blog/blog/2024-10-29-milvus-vs-weaviate/
Docker:Docker部署Milvus向量数据库与可视化界面Attu（对接本地Minio服务）    https://www.cnblogs.com/nhdlb/p/18839349
快速開始使用 Attu Desktop    https://milvus.io/docs/zh-hant/quickstart_with_attu.md
使用Python运行Milvus         https://zhuanlan.zhihu.com/p/617972545
使用 Python 运行Milvus    https://geekdaxue.co/read/milvus-docs-v2/site-zh-CN-getstarted-example_code.md
----------------------------------------------------



16. Weaviate

Weaviate Weaviate 支持混合搜索（向量+关键词）和模块化扩展，适合中小规模数据集。它提供 GraphQL 查询接口，支持水平扩展和多租户架构。
Weaviate 官网说明地址:
https://docs.weaviate.io/deploy/installation-guides/docker-installation       # 在 Docker 中运行 Milvus (Linux)

docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.33.0    # 官方推荐的命令行

docker pull cr.weaviate.io/semitechnologies/weaviate:1.33.0   # 官网显示的连接，但无法下载
docker pull cr.weaviate.io/semitechnologies/weaviate:1.25.0   # 官网显示的连接，但无法下载

docker pull hub.rat.dev/cr.weaviate.io/semitechnologies/weaviate:1.25.0   # 无法下载

docker pull hub.rat.dev/semitechnologies/weaviate:1.25.0    # 可以下载，镜像很小只有，120M大小
docker pull hub.rat.dev/semitechnologies/weaviate:1.33.0    # 可以下载，镜像很小只有，120M大小

docker run -it -d --shm-size=4G --ipc=host --network host -v /Data:/Data hub.rat.dev/semitechnologies/weaviate:1.25.0   # 可以启动，但是，新版的weaviate.Client不支持。
docker run -it -d --shm-size=4G --ipc=host --network host -v /Data:/Data hub.rat.dev/semitechnologies/weaviate:1.33.0   # 使用这个版本的weaviate容器镜像。

docker exec -it 4203bb50eb91f88a082ed22c09e862b7bc705f6137e0b9908097cdb14ae1edac /bin/sh  # Weaviate 的终端是 /bin/sh

# 编写 python 程序， Weaviate_Client_v4.py 测试 Weaviate 的容器启动是否正常:
# 提示:ModuleNotFoundError: No module named 'weaviate'
pip install weaviate-client numpy
pip list |grep -E 'weaviate-client|numpy'
numpy                 2.3.4
weaviate-client       4.17.0

# 简单测试 Weaviate 是否可达:
python3 Weaviate_Client_v4.py

# 测试读写数据:
python3 Weaviate_test_v4_fixed.py

# 以下是加入了 ​​更多测试用例​​ 和 ​​性能计时功能​​ 的 ​​完整程序代码​​，基于您提供的 Weaviate_test_v4_fixed.py 进行了扩展。
# 我们在每个关键操作（如插入、查询、更新、批量操作、向量搜索等）前后添加了 time.time()计时，并输出耗时及相关的性能指标（如每秒操作数、查询响应时间等）。
python3 Weaviate_test_v4_with_performance.py

扩展知识:
----------------------------------------------------
Weaviate 进行容器部署时，部署在GPU上有什么好处？    # 在大语言模型里提问。

Triton + TensorRT Ensemble
演示如何把多个模型（前处理 → 主模型 → 后处理）串起来。

模型热更新与A/B测试
在 Triton 或 vLLM 中热更新权重，做对比测试。

模型压缩与量化
INT8 / FP16 推理。
----------------------------------------------------





17. NVIDIA Nsight Compute 和 NVIDIA Nsight Systems



NVIDIA Nsight Compute，cuda-toolkit 附带的工具。
# 查看Nsight-Compute支持的sections
ncu --list-sections

NVIDIA Nsight Systems，cuda-toolkit 附带的工具。
# 列出当前所有活动的性能分析会话。
nsys sessions list

# 含多种复杂模式:多核核函数、共享内存的矩阵乘法、并行归约、原子操作、流（streams）与异步拷贝、页锁定主机内存、NVTX 区段标注等
# 使用 nsight-systems (nsys) 与 nsight-compute (ncu) 的命令行采样/分析命令（含常用选项、导出 CSV/HTML 的方式）
# 如何从 nsys 时间线与 ncu 性能数据中找出性能瓶颈的具体检查点与示例解释
# 注:下面的程序自足（只依赖 CUDA SDK）。要启用 NVTX 标注，需要 CUDA 的 nvToolsExt（通常随 CUDA toolkit 一起提供）。如果你的环境没有 nvToolsExt，可把 NVTX 相关行注释掉，程序仍能工作。
程序文件:
complex_cuda.cu
complex_cuda_0.cu
matrix_add_profiling.cu 

1. 编译:

# 示例编译（如果要启用 NVTX）:
make NVCCFLAGS="-O3 -arch=sm_80 -lineinfo -DUSE_NVTX"    # 编译不通过。
# 在 nvcr.io/nvidia/pytorch:23.10-py3 容器中的报错:
# 错误提示:tmpxft_00001304_00000000-6_complex_cuda.cudafe1.cpp:(.text.startup+0x1e1): undefined reference to `nvtxRangePushA'

# 在 nvcr.io/nvidia/pytorch:25.09-py3 容器中的报错:
# complex_cuda.cu:9:10: fatal error: nvToolsExt.h: No such file or directory
find /usr -name "nvToolsExt.h" 2>/dev/null

# 使用实际路径替换
nvcc -O3 -arch=sm_80 -lineinfo \
-I/usr/local/cuda-13.0/targets/x86_64-linux/include/nvtx3/nvToolsExt.h \
-L/usr/local/cuda/lib64 \
-DUSE_NVTX -o complex_cuda complex_cuda.cu -lnvToolsExt

# 使用实际路径替换
nvcc -O3 -arch=sm_80 -lineinfo \
-I/usr/local/cuda-13.0/targets/x86_64-linux/include/nvtx3/nvToolsExt.h \
-L/usr/local/cuda/lib64 \
-DUSE_NVTX -o complex_cuda_0 complex_cuda_0.cu -lnvToolsExt


make NVCCFLAGS="-O3 -arch=sm_80 -lineinfo"  # 编译通过，但是缺少 NVTX 。

# nvcc 链接 NVTX 的方式（加上 -lnvToolsExt）:
nvcc -O3 -arch=sm_80 -lineinfo -DUSE_NVTX -o complex_cuda complex_cuda.cu -lnvToolsExt     # 编译通过
nvcc -O3 -arch=sm_80 -lineinfo -DUSE_NVTX -o complex_cuda_0 complex_cuda_0.cu -lnvToolsExt     # 编译通过

-lnvToolsExt:链接 NVTX 库。
-O3:启用优化，使性能分析有意义。
-lineinfo:生成行信息，以便 nsight-compute 可以将指标映射回源代码行。

# 调试，如需进一步排查 libnvToolsExt.so 路径，可以执行:
ls /usr/local/cuda/lib64/libnvToolsExt*

# 常见问题与注意事项:
nsys 与 ncu 都会引入一定的开销，分析结果受影响，请在分析时考虑最佳场景（多次运行取平均）。
ncu --set=full 输出非常多指标；先用 summary / top-k kernels 再深入单个 kernel。
如果程序使用多个进程或 CUDA MPS，会对采集造成不同表现；--target-processes=all 帮助采集子进程。
建议将 -lineinfo 编译选项打开以便 ncu 把指标映射到源码行（Makefile 已加入 -lineinfo）。


2. 使用 nsight-systems (nsys) 命令行分析:

# 基础追踪:
nsys profile --duration <秒数> ./my_app  # 限制追踪时长（避免日志过大）
nsys profile --duration 100 ./complex_cuda

# 生成 .qdrep 报告文件，生成nsys_report.nsys-rep文件:
nsys profile --output <报告文件名> ./my_app  # 保存结果为.nsys-rep文件
# 示例命令（生成 .qdrep 报告文件）:
# 基础采集命令（对整个进程）:
nsys profile --output=nsys_report --trace=cuda,osrt,nvtx --sample=cpu ./complex_cuda    # 得到同目录下， nsys_report.nsys-rep 文件。

# 导出火焰图 / timeline 转成 HTML（更直观）:
nsys-ui nsys_report.qdrep
# 或者导出 timeline: (需要 GUI 通常)


# 导出概要（文本）:
nsys stats --report=summary nsys_report.nsys-rep > nsys_summary.txt      # 得到同目录下， nsys_summary.txtp 文件。

扩展知识:
----------------------------------------------------
在GPU节点上，安装 cuda_11.8.0_520.61.05_linux.run ，文件会安装在 /usr/local/cuda-11.8/ 目录中，
在 /usr/local/cuda-11.8/nsight-compute-2022.3.0/ 中，有 ncu ncu-ui nv-nsight-cu nv-nsight-cu-cli 命令行工具。
在 /usr/local/cuda-11.8/nsight-systems-2022.4.2/ 中，有 nsys nsys-ui 命令行工具。

在 nvcr.io/nvidia/pytorch:23.10-py3容器中:
在 /opt/nvidia/nsight-compute/2023.2.2/中，有 ncu ncu-ui 命令行工具。容器中所带的命令行工具，比 run 文件安装的要少。
在 /etc/alternatives/cuda-12/NsightSystems-cli-2023.3.1/ 中，有 nsys 命令行工具。

在 nvcr.io/nvidia/pytorch:25.09-py3 容器中:
在 /opt/nvidia/nsight-compute/2025.3.1 中，有 ncu ncu-ui 命令行工具。容器中所带的命令行工具，比 run 文件安装的要少。
在 /usr/local/cuda-13.0/NsightSystems-cli-2025.5.1/target-linux-x64/ 中，有 nsys 命令行工具。
----------------------------------------------------



3. 使用 nsight-compute (ncu) 命令行分析:
通过 NVIDIA Nsight Compute (ncu) 分析 GPU 指令流效率（Instruction Flow Efficiency） 时，关键是理解指令执行的瓶颈来源、SM（Streaming Multiprocessor）利用率以及分支、调度和依赖关系等影响因素。

# 导出 CSV（只示例几个重要 metric）:
ncu -o ncu_report_csv --csv --metrics achieved_occupancy,sm_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput,inst_executed ./complex_cuda > metrics.csv
ncu -o ncu_report_csv --csv --metrics achieved_occupancy,sm_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput,inst_executed ./complex_cuda_0 > metrics.csv

# 导出 HTML:
ncu --export report.html ./complex_cuda
ncu --export report.html ./complex_cuda_0

# 基本性能分析
ncu -o profile --metrics smsp__cycles_active.avg.pct_of_peak_sustained ./complex_cuda
ncu -o profile --metrics smsp__cycles_active.avg.pct_of_peak_sustained ./complex_cuda_0

ncu --set=full --target-processes=all -o ncu_report ./complex_cuda
ncu --set full --target-processes all -o ncu_report ./complex_cuda

# 只采集指令流相关指标：
ncu --metrics sm__inst_executed.sum,sm__inst_executed_per_issue_active.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.avg.pct ./complex_cuda

# 生成 .ncu-rep 文件后可在 GUI 中打开：
ncu-ui report.ncu-rep



4. openmpi 测试分析程序:
# openmpi 参考文档:
https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man1/mpirun.1.html#


mpirun --allow-run-as-root -np 2 --bind-to core ./complex_cuda
mpirun --allow-run-as-root -np 2 --bind-to core -H aa,aa,bb ./complex_cuda

mpirun --np 4 --report-bindings --map-by core --bind-to core --allow-run-as-root ./complex_cuda
mpirun --np 4 --report-bindings --map-by core --bind-to core --allow-run-as-root ./complex_cuda_0
mpirun --np 4 --report-bindings --map-by core --bind-to core --allow-run-as-root ./matrix_add_profiling

mpirun --allow-run-as-root --np 4 --report-bindings --map-by slot:PE=2 --use-hwthread-cpus ./matrix_add_profilin


扩展知识:
----------------------------------------------------
在 gemini 提问:
编写一个最复杂的，最全面的CUDA程序，要求带偏置的矩阵加法 kernel 和测试，
每个 kernel 用 NVTX 分区包裹，方便分析，可根据需要扩展更多测试，如不同 stream、不同 block size、不同数据规模等。
并使用nsight-compute和nsight-systems的命令行工具分析这个程序的运行过程。

matrix_add_profiling.cu   # 大语言模型输出程序文件

nvcc -o matrix_add_profiling matrix_add_profiling.cu -O3 -lineinfo -lnvToolsExt

# 需要增加编译参数:
nvcc -o matrix_add_profiling matrix_add_profiling.cu -O3 -lineinf -lnvToolsExt -gencode arch=compute_70,code=sm_70   # 指定 GPU 架构，且指定正确的 GPU 架构

为什么这个命令有效？
-gencode arch=compute_70,code=sm_70:
arch=compute_70:告诉编译器生成计算能力 7.0 的 PTX（一种中间代码，用于未来的 JIT 编译）。
code=sm_70:告诉编译器直接生成计算能力 7.0 的本机二进制代码 (SASS)。

nvcc -O3 -arch=sm_80 -lineinfo -DUSE_NVTX

----------------------------------------------------




扩展知识:
----------------------------------------------------
docker pull nvcr.io/nvidia/pytorch:25.09-py3   # "Ubuntu 24.04.3 LTS"
docker run -it -d --gpus all --cap-add SYS_ADMIN --net=host --pid=host --ipc=host --privileged -v /Data:/Data nvcr.io/nvidia/pytorch:25.09-py3 


docker pull pytorch/manylinux2_28-builder:cuda12.9
docker pull hub.rat.dev/pytorch/manylinux2_28-builder:cuda12.9   # 使用代理才能 pull 下来
docker run -it -d --gpus all --cap-add SYS_ADMIN --net=host --pid=host --ipc=host --privileged -v /Data:/Data hub.rat.dev/pytorch/manylinux2_28-builder:cuda12.9

AlmaLinux release 8.10 
Linux anhua209 5.15.0-153-generic #163-Ubuntu SMP Thu Aug 7 16:37:18 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux
----------------------------------------------------



# 还未整理:


ResNet50

14. 分布式训练优化与常见问题排查（NCCL调优、AMP、FSDP）
1. NCCL调优
NCCL基础配置优化
NCCL性能监控
2. Automatic Mixed Precision (AMP) 优化
AMP基础使用
AMP性能监控
3. FSDP (Fully Sharded Data Parallel) 优化
FSDP基础配置
FSDP性能优化
4. 常见问题排查
内存相关问题

ResNet50

15. TensorRT 部署与 Triton 集成（加速 Stable Diffusion 等）


❌🎉✅🔍📄ℹ️📚🚀📊🎉🚪📦📊 🔧 🧪📢📈🐛🏡   ⛵️


```
