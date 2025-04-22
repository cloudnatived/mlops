
### 2.1.Nvidia驱动和CUDA toolkit

apt方式，Ubuntu22.04安装Nvidia 550驱动和CUDA toolkit 12.4.1（包括CUDA、NCCL）

```
1. 安装显卡驱动550版本：
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get -y install cuda-drivers

命令来源于：
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network
安装完成后重启，重启之后可以输入nvidia-smi命令验证：nvidia-smi 

2. 安装CUDA toolkit 12.4.1（包括CUDA、NCCL）：
apt-get -y install cuda-toolkit-12-4

3. 为CUDA12.4在.bashrc中添加环境变量
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

4. 验证CUDA toolkit 12.4.1安装成功
xiaxinkai@msi:~$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0

5、卸载驱动库
apt-get --purge remove nvidia*
apt autoremove

卸载cuda
dpkg -l | grep nvidia

remove CUDA Toolkit:
apt-get --purge remove "*cublas*" "cuda*"

卸载依赖文件
apt-get --purge remove "*nvidia*"  （也可用：NVIDIA-Linux-x86_64-465.31.run --uninstall卸载，大同小异。）

检查是否卸载彻底
dpkg -l | grep nvidia（如果卸载干净了，这条指令后将无提示。）

```



### 2.2.Nvidia驱动和CUDA toolkit的run文件安装

在nvidia.com选择合适版本的Nvidia的driver的run文件安装（（包括驱动、CUDA、NCCL））。

```
https://www.nvidia.cn/drivers/lookup/
wget https://cn.download.nvidia.com/tesla/550.144.03/nvidia-driver-local-repo-ubuntu2204-550.144.03_1.0-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run

cuda_12.4.1_550.54.15_linux.run
┌─┐
│ CUDA Installer se Agreement                                                  │
│ - [X] Driver                                                                 │
│      [X] 550.54.15                                                           │
│ + [X] CUDA Toolkit 12.4                                                      │
│   [X] CUDA Demo Suite 12.4                                                   │
│   [X] CUDA Documentation 12.4                                                │
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
│                                                                              
│                                                                              │
│ Up/Down: Move | Left/Right: Expand | 'Enter': Select | 'A': Advanced options │
└─┘


root@y249:/Data# ./cuda_12.4.1_550.54.15_linux.run 
===========
= Summary =
===========

Driver:   Installed
Toolkit:  Installed in /usr/local/cuda-12.4/

Please make sure that
 -   PATH includes /usr/local/cuda-12.4/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-12.4/lib64, or, add /usr/local/cuda-12.4/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-12.4/bin
To uninstall the NVIDIA Driver, run nvidia-uninstall
Logfile is /var/log/cuda-installer.log

cat >> /etc/profile << EOF
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOF

# 工作中碰到的实际问题。
4090卡最低驱动CUDA-12.8，H200最低驱动CUDA-12.6。但是在CUDA-12.6中，有一些NCCL的通讯支持有各种问题。
```




CUDA (Compute Unified Device Architecture)

```
CUDA(Compute Unified Device Architecture)：是一种由 NVIDIA 推出的通用并行计算架构，该架构使 GPU 能够解决复杂的计算问题;，它包括编译器(nvcc)、开发工具、运行时库和驱动等模块，是当今最流行的GPU编程环境
cuDNN：是基于 CUDA 的深度学习 GPU 加速库，支持常见的深度学习计算类型(卷积、下采样、非线性、Softmax 等)

一个基本的 CUDA 程序架构包含 5 个主要方面：
分配 GPU 内存
复制 CPU内存数据到 GPU 内存
激活 CUDA 内核去执行特定程序的计算
将数据从 GPU 拷贝 到 CPU 中
删除 GPU 中的数据
```





```
# 1、查看当前 cuda 版本
nvcc -V 
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation
Built on Fri_Sep__1_21:08:03_CDT_2017
Cuda compilation tools, release 9.0, V9.0.176

# 2、删除之前创建的软链接
sudo rm -rf /usr/local/cuda

# 3、建立新的软链接，cuda9.0 切换到 cuda11.0
sudo ln -s /usr/local/cuda-11.0/ /usr/local/cuda/

# 4、查看当前 cuda 版本
nvcc -V 
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Wed_Jul_22_19:09:09_PDT_2020
Cuda compilation tools, release 11.0, V11.0.221
Build cuda_11.0_bu.TC445_37.28845127_0

# 5、将 ~/.bashrc 或 caffe Makefile.config 等下与 cuda 相关的路径都改为 /usr/local/cuda/（指定版本的软链接）
vim ~/.bashrc  # 不使用 /usr/local/cuda-9.0/ 或 /usr/local/cuda-11.0/ 等，这样每次切换版本，就不用改配置了

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64

source ~/.bashrc  # 立即生效

# cuda-12.4的路径。
/usr/local/cuda-12.4/

# 下载cuda-samples的对应版本v12.4.1
https://github.com/NVIDIA/cuda-samples/releases/tag/v12.4.1
https://github.com/NVIDIA/cuda-samples/archive/refs/tags/v12.4.1.tar.gz

# cuda-samples的v12.4.1编译。
make

# 运行示例
Samples/0_Introduction/asyncAPI/asyncAPI 
Samples/0_Introduction/cudaOpenMP/cudaOpenMP
Samples/0_Introduction/matrixMul/matrixMul

1. 实用工具：实用工具示例，演示如何查询设备功能并测量 GPU/CPU 带宽。
2. 概念与技术：演示 CUDA 相关概念和常用问题解决技术的示例。
3. CUDA 功能：演示 CUDA 功能的示例（协同组、CUDA 动态并行、CUDA 图形等）。
4. CUDA 库：演示如何使用 CUDA 平台库（NPP、NVJPEG、NVGRAPH cuBLAS、cuFFT、cuSPARSE、cuSOLVER 和 cuRAND）的示例。
5. 领域特定：特定于领域的示例（图形、金融、图像处理）。
6. 性能：演示性能优化的示例。
7. libNVVM：演示如何使用 libNVVVM 和 NVVM IR 的示例。

```
