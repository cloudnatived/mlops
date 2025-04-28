https://github.com/ifromeast/cuda_learning/tree/main

# cuda_learning
learning how CUDA works

# project list:
- custom op [Done]
    - [CUDA 编程基础]
- memory & reduction [Done]
    - [GPU的内存体系及其优化指南]
- Gemm [Done]
    - [通用矩阵乘法：从入门到熟练]
- Transformer [Done]
    - 基础算子：
        - [LayerNorm 算子的 CUDA 实现与优化]
        - [SoftMax 算子的 CUDA 实现与优化]
        - [Cross Entropy 的 CUDA 实现]
        - [AdamW 优化器的 CUDA 实现]
        - [激活函数与残差连接的 CUDA 实现]
        - [embedding 层与 LM head 层的 CUDA 实现]
    - 核心模块
        - [self-attention 的 CUDA 实现及优化 (上)]
        - [self-attention 的 CUDA 实现及优化 (下)]
    
- CUDA mode lectures 
- DeepSeek infra cases


## CUDA（一）：CUDA 编程基础
## 01_cuda_op/  
实践：PyTorch自定义CUDA算子，pytorch构造CUDA算子库扩展。

算子构建
实现如下所示，其中 MatAdd 是 kernel 函数，运行在GPU端，而launch_add2是CPU端的执行函数，调用kernel，它是异步的，调用完之后控制权立刻返回给CPU。
kernel/add2_kernel.cu
```
__global__ void MatAdd(float* c,
                            const float* a,
                            const float* b,
                            int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = j*n + i;
    if (i < n && j < n)
        c[idx] = a[idx] + b[idx];
}

void launch_add2(float* c,
                 const float* a,
                 const float* b,
                 int n) {
    dim3 block(16, 16);
    dim3 grid(n/block.x, n/block.y);

    MatAdd<<<grid, block>>>(c, a, b, n);
}
```

Torch C++ 封装  
CUDA 的 kernel 函数 PyTorch 并不能直接调用，还需要提供一个接口，这个功能在 add2_ops.cpp 中实现
kernel/add2_ops.cpp
```
#include <torch/extension.h>
#include "add2.h"

void torch_launch_add2(torch::Tensor &c,
                       const torch::Tensor &a,
                       const torch::Tensor &b,
                       int64_t n) {
    launch_add2((float *)c.data_ptr(),
                (const float *)a.data_ptr(),
                (const float *)b.data_ptr(),
                n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_add2",
          &torch_launch_add2,
          "add2 kernel warpper");
}

TORCH_LIBRARY(add2, m) {
    m.def("torch_launch_add2", torch_launch_add2);
} 
```
torch_launch_add2函数传入的是C++版本的torch tensor，然后转换成C++指针数组，调用CUDA函数launch_add2来执行核函数。这里用 pybind11 来对torch_launch_add2函数进行封装，然后用cmake编译就可以产生python可以调用的.so库。

Torch 使用CUDA 算子 主要分为三个步骤：

先编写CUDA算子和对应的调用函数。
然后编写torch cpp函数建立PyTorch和CUDA之间的联系，用pybind11封装。
最后用PyTorch的cpp扩展库进行编译和调用。

编译及调用方法  
JIT 编译调用

just-in-time(JIT, 即时编译)，即 python 代码运行的时候再去编译cpp和cuda文件。

首先需要加载需要即时编译的文件，然后调用接口函数
```
from torch.utils.cpp_extension import load
cuda_module = load(name="add2",
                           extra_include_paths=["include"],
                           sources=["kernel/add2_ops.cpp", "kernel/add2_kernel.cu"],
                           verbose=True)
```


```
python3 -c "import torch; print(torch.cuda.get_device_capability())"
export TORCH_CUDA_ARCH_LIST="8.9"

编译：
python3 setup.py install
执行：
python3 run_time.py --compiler setup
```


CMAKE 编译调用  
最后就是cmake编译的方式了，要编写一个CMakeLists.txt文件，需要关注的几个点在于：依赖库的匹配、编译过程及软连接的建立。

```
/etc/profile 添加环境变量
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64

export C_INCLUDE_PATH=/usr/include/python3.10:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/usr/include/python3.10:$CPLUS_INCLUDE_PATH



//cpp端用的是TORCH_LIBRARY进行封装：
TORCH_LIBRARY(add2, m) {
    m.def("torch_launch_add2", torch_launch_add2);
}

//最后会在build目录下生成一个libadd2.so，通过如下方式在python端调用：
import torch
torch.ops.load_library("build/libadd2.so")
torch.ops.add2.torch_launch_add2(c, a, b, n)


编译：
mkdir build;
cd build;
cmake ..;
make;

执行：
python3 run_time.py --compiler cmake
```

# CUDA（二）：GPU的内存体系及其优化指南
```
3 种方式都完成后，我们可以编译运行代码

nvcc reduce_gpu.cu -o reduce
通过计时函数可以看到，每种方法的完整计算总时间都在 7.5 ms 左右。
然后通过 nvprof 命令查看GPU各部分的耗时
nvprof ./reduce

如果要放大访存速度的差别，可以使用双精度，编译方式如下：
nvcc reduce_gpu.cu -DUSE_DP -o reduce_dp
此时可以看到全局内存的性能出现了明显下降：



```

# CUDA（三）：通用矩阵乘法，从入门到熟练


# CUDA（四）：使用 CUDA 实现 Transformer 结构
```
python prepro_tinyshakespeare.py

2.2.1 训练过程的实现
更多的细节请参考github.com/karpathy/llm.c/blob/master/train_gpt2.cu，编译及运行如下：
make train_gpt2cu
./train_gpt2cu

2.2.2 使用 cuDNN 模块
为了获得更好的性能，接下来使用 cuDNN 模块，编译及运行命令如下：
make train_gpt2cu USE_CUDNN=1
./train_gpt2cu


```
