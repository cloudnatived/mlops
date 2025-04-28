
# CUDA
CUDA(Compute Unified Device Architecture)：是一种由 NVIDIA 推出的通用并行计算架构，该架构使 GPU 能够解决复杂的计算问题;，它包括编译器(nvcc)、开发工具、运行时库和驱动等模块，是当今最流行的GPU编程环境
cuDNN：是基于 CUDA 的深度学习 GPU 加速库，支持常见的深度学习计算类型(卷积、下采样、非线性、Softmax 等)

一个基本的 CUDA 程序架构包含 5 个主要方面：
分配 GPU 内存
复制 CPU内存数据到 GPU 内存
激活 CUDA 内核去执行特定程序的计算
将数据从 GPU 拷贝 到 CPU 中
删除 GPU 中的数据

## 一. Nvidia卡驱动和CUDA toolkit
### apt方式，Ubuntu22.04安装Nvidia 550驱动和CUDA toolkit 12.4.1（包括CUDA、NCCL）

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



### Nvidia驱动和CUDA toolkit的run文件安装
### 在nvidia.com选择合适版本的Nvidia的driver的run文件安装（（包括驱动、CUDA、NCCL））

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

### CUDA常用操作命令
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

```


    
## 二. CUDA toolkit中的命令行工具
### CUDA自带的命令行工具
### NVIDIA NVIDIA-smi
**NVIDIA-smi**：这是NVIDIA提供的命令行工具，用于监控GPU的实时状态。它能够显示GPU利用率、显存使用情况、温度、功耗等关键指标。
**DCGM（Data Center GPU Manager）**：对于数据中心环境，DCGM提供了更高级的监控和管理功能，支持大规模GPU集群的监控和自动化管理。
```
GPU利用率:
指标含义：GPU利用率表示GPU在单位时间内实际用于计算的时间比例。高利用率意味着GPU资源被充分利用，而低利用率可能表示资源浪费或计算任务未充分利用GPU的并行能力。
nvidia-smi --query-gpu=utilization.gpu --format=csv

显存利用率:
指标含义：显存利用率反映了GPU显存被占用的情况。如果显存利用率过高，可能会导致GPU频繁进行内存交换，降低计算性能。
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

计算吞吐量:
指标含义：计算吞吐量表示GPU在单位时间内能够完成的计算量。例如，在图像分类任务中，计算吞吐量可以表示为每秒处理的图像数量。
监控方法：可以通过AI框架（如TensorFlow、PyTorch）提供的日志或性能分析工具来测量计算吞吐量。

指标含义：GPU在高负载运行时会产生大量热量。监控GPU温度可以防止GPU过热而损坏。一般来说，GPU的温度应保持在安全范围内（通常在30 - 50摄氏度之间）。
nvidia-smi --query-gpu=temperature.gpu --format=csv

指标含义：GPU的功耗是一个重要的监控指标。一方面，要确保GPU的功耗在硬件允许的范围内，避免电源供应不足导致的硬件故障。另一方面，从能源成本角度考虑，合理控制GPU功耗也很重要。
nvidia-smi --query-gpu=power.draw --format=csv

实时监控GPU状态:
说明：watch命令用于实时监控，-n 1表示每秒刷新一次。nvidia-smi命令显示GPU的实时状态，包括利用率、显存使用情况、温度和功耗等。
watch -n 1 "nvidia-smi"

详细监控GPU性能:
说明：--query-gpu参数用于指定要查询的指标，--format=csv输出为CSV格式，方便后续分析。-l 1表示每秒记录一次数据。
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,power.draw,temperature.gpu --format=csv -l 5

监控特定GPU:
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -i 0

使用DCGM进行高级监控:
说明：dcgmi discovery -l列出所有可用的GPU和它们的状态。dcgmi dmon -e 1001 -d 1启动监控，-e 1001表示监控所有指标，-d 1表示每秒记录一次数据。
dcgmi discovery -l
dcgmi dmon -e 1001 -d 1
```
    
### NVIDIA Nsight Compute
```
Nsight系列工具中的一个组件，专门用于CUDA核函数的性能分析，它是更接近内核的分析。它允许开发人员对 CUDA 核函数进行详细的性能分析，包括核函数的时间分布、内存访问模式、并行性、指令分发等。Nsight Compute提供了许多有用的数据和图形化的界面，帮助开发人员深入理解和优化核函数的性能。
ncu命令主要分析kernel内部的带宽、活跃线程数、warp利用率等。
地址：https://developer.nvidia.com/nsight-compute
DOC：https://developer.nvidia.com/tools-overview/nsight-compute/get-started

nsight-compute包含在cuda的目录里。
/usr/local/cuda-12.4/nsight-compute-2024.1.1/

比较常用的分析：
①核函数的roofline分析，从而知道核函数是计算密集型还是访存密集型；
②occupancy analysis：对核函数的各个指标进行估算一个warp的占有率的变化；
③memory bindwidth analysis 针对核函数中对各个memory的数据传输带宽进行分析，可以比较好的理解memory架构；
④shared memory analysis：针对核函数中对shared memory访问以及使用效率进行分析；

/usr/local/cuda-12.4/nsight-compute-2024.1.1/
# 详细介绍。
https://docs.nvidia.com/nsight-compute/2024.3/CustomizationGuide/index.html

带宽查看有两个指标，分别是global memory的带宽和dram的带宽，global memory的带宽指标其实指的是L2Cache和L1Cache到SM的带宽，因为SM寻找数据会先去Cache寻找，Cache找不到再去GPU的DRAM中，所以有的时候会发现global memory的带宽会高于英伟达给的带宽参数。而DRAM带宽就是对应英伟达官方给的带宽。

命令行工具是/usr/local/cuda-12.4/nsight-compute-2024.1.1目录下的：ncu ncu-ui

# 查看Nsight-Compute支持的sections
ncu --list-sections

# 获取所有的metrics
ncu --set full --export ncu_report -f ./sample_2

# 获取指定section
ncu --section ComputeWorkloadAnalysis --print-details all ./sample_2

# 获取指定section和特定的metrics
ncu --section WarpStateStats --metrics smsp__pcsamp_sample_count,group:smsp__pcsamp_warp_stall_reasons,group:smsp__pcsamp_warp_stall_reasons_not_issued ./sample_2

ncu --metrics group:smsp__pcsamp_warp_stall_reasons ./sample_2
ncu --metrics group:smsp__pcsamp_warp_stall_reasons_not_issued ./sample_2
ncu --metrics group:memory__dram_table ./sample_2

ncu --metrics gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,breakdown:gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed  ./sample_2
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,breakdown:sm__throughput.avg.pct_of_peak_sustained_elapsed ./sample_2
ncu --metrics l1tex__lsuin_requests,l1tex__f_wavefronts,dram__sectors,dramc__sectors,fbpa__dram_sectors,l1tex__data_bank_reads,l1tex__data_pipe_lsu_wavefronts,l1tex__lsuin_requests,l1tex__t_bytes_lookup_miss  ./sample_2

ncu --metrics sass__inst_executed_per_opcode,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active --target-processes all --export ncu_report -f ./sample_2

# 查看instances的数据
ncu -i ncu_report.ncu-rep --print-details all --print-units auto --print-metric-instances values

参考资料：
NVIDIA性能分析工具nsight-compute入门    https://zhuanlan.zhihu.com/p/662012270
NsightComputeProfiling入门    https://blog.csdn.net/m0_61864577/article/details/140618800

```

### NVIDIA Nsight Systems
```
系统级的性能分析工具，用于分析和优化整个CUDA应用程序或系统的性能。它可以提供对应用程序整体性能的全面见解，以及考察GPU活动、内存使用、线程间通信等方面的详细信息，它提供了可视化界面和统计数据，开发人员可以使用它来发现性能瓶颈、调整应用程序的配置，以及提高整体性能。
nsys命令主要分析api级别的性能时间等。
地址：https://developer.nvidia.com/nsight-systems
DOC：https://developer.nvidia.com/nsight-systems/get-started

# nsight-compute包含在cuda的目录里。

/usr/local/cuda-12.4/nsight-systems-2023.4.4/
命令行工具为/usr/local/cuda-12.4/nsight-systems-2023.4.4/bin下的nsys，nsys-ui。

nsys 提供了强大的命令行界面（CLI），方便用户进行各种性能分析操作。以下是一些最常用的命令及其功能：

nsys profile [options] [application] [application args]: 这是最核心的命令，用于启动应用程序并捕获其性能数据。

--trace=<trace>: 通过这个选项，你可以指定要跟踪的 API 或事件类型，例如 cuda（跟踪 CUDA API 和内核）、cudart（跟踪 CUDA 运行时 API）、osrt（跟踪操作系统运行时 API）、opengl、vulkan 等。你可以使用逗号分隔多个跟踪类型，例如 --trace=cuda,osrt,vulkan。
-o <filename>: 使用此选项指定输出报告文件的名称，通常以 .qdrep 格式保存。例如，-o my_report.qdrep。
--duration=<seconds>: 设置性能分析的持续时间，单位为秒。例如，--duration=10 将会分析应用程序运行的 10 秒。
--delay=<seconds>: 设置开始性能分析前的延迟时间，单位为秒。这在需要等待应用程序启动完成后再开始分析时非常有用。
--gpu-metrics-device=<device_id>: 如果你的系统中有多个 GPU，可以使用此选项指定要收集 GPU 指标的设备 ID。
--cudabacktrace=all|api|kernel|none: 控制 CUDA 回溯信息的收集级别，有助于更深入地了解 CUDA API 调用和内核执行的上下文。
想要了解更多选项，请随时使用 nsys profile --help 命令查看完整的帮助文档。
nsys launch [options] [application] [application args]: 这个命令用于启动应用程序，并使其处于等待性能分析器连接的状态。这在你需要从另一个进程或机器上连接 nsys 进行分析时非常有用。

nsys start [options]: 启动一个新的性能分析会话。通常会与 nsys stop 命令配合使用，用于在应用程序运行的特定时间段内进行性能分析。
--trace=<trace>: 同样用于指定要跟踪的 API 或事件类型。
--output=<filename>: 指定输出报告文件的名称。

nsys stop: 停止当前正在运行的性能分析会话，并将收集到的性能数据保存到指定的文件中。
nsys cancel: 如果你想要放弃当前的性能分析会话，可以使用这个命令取消并丢弃已经收集到的数据。
nsys service: 启动 Nsight Systems 数据服务，这是 nsys 工具后台运行的一个重要组成部分。
nsys stats <filename>.qdrep: 这个命令可以从一个已经存在的 .qdrep 报告文件中生成各种统计信息，帮助你快速了解性能概况。
nsys status: 显示当前 nsys 的运行状态，例如是否有正在进行的性能分析会话。
nsys shutdown: 关闭所有与 nsys 相关的进程。
nsys sessions list: 列出当前所有活动的性能分析会话。
nsys export <filename>.qdrep --type=<format> -o <output_filename>: 将 .qdrep 报告文件导出为其他格式，例如 csv（逗号分隔值）、sqlite（SQLite 数据库）等，方便与其他工具进行数据交换和分析。
nsys analyze <filename>.qdrep: 分析报告文件，nsys 可能会识别出潜在的性能优化机会并给出建议。
nsys recipe <recipe_file>: 用于运行多节点分析的配方文件，这对于分析分布式应用程序非常有用。
nsys nvprof [nvprof options] [application] [application args]: 对于熟悉 NVIDIA 之前的性能分析工具 nvprof 的用户，nsys 提供了这个命令来尝试将 nvprof 的命令行选项转换为 nsys 的选项并执行分析，方便用户进行迁移。

要获取任何特定命令的更详细信息，只需在终端中运行 nsys --help <command> 即可。例如，要查看 nsys profile 命令的所有可用选项，可以执行 nsys --help profile。


比较常用的分析：
①对kernel执行和memory进行timeline分析，尝试优化,隐藏memory access或者删除冗长的memory access；多流调度，融合kernel减少kernel launch的overhead；CPU与GPU的overlapping；
②分析DRAM以及PCIe带宽的使用率，没有使用shared memory，那么DRAM brandwidth就没有变化，可以分析哪些带宽没有被充分利用；
③分析warp的占有率，从而可以知道一个SM中计算资源是否被充分利用；

NVIDIA Nsight Systems (nsys) 是一款功能强大的系统级性能分析工具，它通过提供全方位的性能数据和直观的可视化界面，帮助开发者深入了解应用程序在整个系统中的行为。掌握 nsys 的使用，能够让你更有效地识别性能瓶颈，优化资源利用率，最终提升应用程序的整体效率。

参考资料：
cuda学习日记(6) nsight system / nsight compute    https://zhuanlan.zhihu.com/p/640344249
NVIDIA Nsight Systems (nsys) 工具使用    https://www.cnblogs.com/menkeyi/p/18791669
```


## 三. cuda-samples
cuda-samples每个版本都与CUDA toolkit对应  
```
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


## 四. NVIDIA CUDA Library Samples

NVIDIA CUDA Library Samples 项目是由 NVIDIA 公司开发并开源的，旨在展示如何使用 GPU 加速库进行高性能计算。这些库包括数学运算、图像处理、信号处理、线性代数和压缩等多个领域。项目中的示例代码展示了如何利用这些库来加速各种计算任务。    
主要的编程语言是 C++ 和 CUDA C/C++，因为这些语言能够充分利用 GPU 的并行计算能力。    

The CUDA Library Samples repository contains various examples that demonstrate the use of GPU-accelerated libraries in CUDA. These libraries enable high-performance computing in a wide range of applications, including math operations, image processing, signal processing, linear algebra, and compression. The samples included cover:

    Math and Image Processing Libraries
    cuBLAS (Basic Linear Algebra Subprograms)
    cuTENSOR (Tensor Linear Algebra)
    cuSPARSE (Sparse Matrix Operations)
    cuSOLVER (Dense and Sparse Solvers)
    cuFFT (Fast Fourier Transform)
    cuRAND (Random Number Generation)
    NPP (Image and Video Processing)
    nvJPEG (JPEG Encode/Decode)
    nvCOMP (Data Compression)

Library Examples

Explore the examples of each CUDA library included in this repository:

    cuBLAS - GPU-accelerated basic linear algebra (BLAS) library
    cuBLASLt - Lightweight BLAS library
    cuBLASMp - Multi-process BLAS library
    cuBLASDx - Device-side BLAS extensions
    cuDSS - GPU-accelerated linear solvers
    cuFFT - Fast Fourier Transforms
    cuFFTMp - Multi-process FFT
    cuFFTDx - Device-side FFT extensions
    cuPQC - Post-Quantum Cryptography device library
    cuRAND - Random number generation
    cuSOLVER - Dense and sparse direct solvers
    cuSOLVERMp - Multi-process solvers
    cuSOLVERSp2cuDSS - Transition example from cuSOLVERSp/Rf to cuDSS
    cuSPARSE - BLAS for sparse matrices
    cuSPARSELt - Lightweight BLAS for sparse matrices
    cuTENSOR - Tensor linear algebra library
    cuTENSORMg - Multi-GPU tensor linear algebra
    NPP - GPU-accelerated image, video, and signal processing functions
    NPP+ - C++ extensions for NPP
    nvJPEG - High-performance JPEG encode/decode
    nvJPEG2000 - JPEG2000 encoding/decoding
    nvTIFF - TIFF encoding/decoding
    nvCOMP - Data compression and decompression

Each sample provides a practical use case for how to apply these libraries in real-world scenarios, showcasing the power and flexibility of CUDA for a wide variety of computational needs.
Additional Resources

For more information and documentation on CUDA libraries, please visit:

    CUDA Toolkit Documentation
    NVIDIA Developer Zone
    CUDA Samples


    

参考资料：

OpenCL安装过程记录    https://zhuanlan.zhihu.com/p/615906584    
一文搞懂OpenCL    https://blog.csdn.net/RuanJian_GC/article/details/132570906    
CUDA Library Samples 使用教程    https://blog.csdn.net/gitblog_00134/article/details/142774402
