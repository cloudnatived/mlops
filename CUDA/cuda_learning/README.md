# cuda_learning
learning how CUDA works

## project list:
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


CUDA（一）：CUDA 编程基础
实践：PyTorch自定义CUDA算子.

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
编译：
python3 setup.py install
执行：
python run_time.py --compiler setup
```


CMAKE 编译调用
最后就是cmake编译的方式了，要编写一个CMakeLists.txt文件，需要关注的几个点在于：依赖库的匹配、编译过程及软连接的建立。

```
//cpp端用的是TORCH_LIBRARY进行封装：
TORCH_LIBRARY(add2, m) {
    m.def("torch_launch_add2", torch_launch_add2);
}

//最后会在build目录下生成一个libadd2.so，通过如下方式在python端调用：
import torch
torch.ops.load_library("build/libadd2.so")
torch.ops.add2.torch_launch_add2(c, a, b, n)


编译：
mkdir build
cd build
cmake ..
make

执行：
python3 run_time.py --compiler cmake
```
