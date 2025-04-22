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
