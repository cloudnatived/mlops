# cuda-samples
## https://github.com/NVIDIA/cuda-samples/


```
0. Introduction
Basic CUDA samples for beginners that illustrate key concepts with using CUDA and CUDA runtime APIs.
面向初学者的基本 CUDA 示例，讲解使用 CUDA 和 CUDA 运行时 API 的关键概念。
这些示例展示了 CUDA 编程的各种基本和高级技术，从简单的算术运算到复杂的并行计算和优化策略，为用户提供了丰富的学习和实践资源。

1. Utilities
Utility samples that demonstrate how to query device capabilities and measure GPU/CPU bandwidth.
这些实用程序示例演示了如何查询设备功能以及测量 GPU/CPU 带宽。
这些实用工具示例为用户提供了方便的方法来测量和查询系统中 CUDA 设备的性能和属性，有助于优化 CUDA 应用程序的性能和资源利用。

2. Concepts and Techniques
Samples that demonstrate CUDA related concepts and common problem solving techniques.
这些示例演示了 CUDA 相关概念和常见的问题解决技术。
概念和技术。此部分的示例展示了与 CUDA 相关的概念以及解决常见问题的方法。例如，如何有效地管理内存、优化线程调度、处理并行计算中的常见挑战等。

3. CUDA Features
Samples that demonstrate CUDA Features (Cooperative Groups, CUDA Dynamic Parallelism, CUDA Graphs etc).
这些示例演示了 CUDA 特性（协作组、CUDA 动态并行、CUDA 图等）。
这些示例展示了 CUDA 的一些高级功能，如张量核心、动态并行、图形 API 等，帮助用户了解和利用这些功能来提高计算性能和效率。

4. CUDA Libraries
Samples that demonstrate how to use CUDA platform libraries (NPP, NVJPEG, NVGRAPH cuBLAS, cuFFT, cuSPARSE, cuSOLVER and cuRAND).
这些示例演示了如何使用 CUDA 平台库（NPP、NVJPEG、NVGRAPH、cuBLAS、cuFFT、cuSPARSE、cuSOLVER 和 cuRAND）。
这些示例展示了如何使用 CUDA 平台库进行各种高级计算任务，从线性代数到图像处理和随机数生成，帮助用户了解和使用这些库来提高其 CUDA 应用程序的性能和功能。

5. Domain Specific
Samples that are specific to domain (Graphics, Finance, Image Processing).
这些示例针对特定领域（图形、金融、图像处理）。
这些示例展示了 CUDA 在图像处理、金融模拟、物理仿真等领域的应用，帮助用户了解如何在特定应用场景中利用 CUDA 技术提高性能和效率。

6. Performance
Samples that demonstrate performance optimization.
这些示例演示了性能优化。
这些示例展示了在 CUDA 编程中如何通过优化内存对齐、选择合适的内存类型和传输方式来提高数据传输和计算的性能，从而实现高效的 GPU 编程。

7. libNVVM
Samples that demonstrate the use of libNVVVM and NVVM IR.
演示 libNVVVM 和 NVVM IR 用法的示例。
这里的示例展示了如何使用 libNVVM 和 NVVM IR（NVIDIA CUDA Compiler Intermediate Representation）。这些工具用于高级的 CUDA 编译和优化，通过理解和使用这些工具，用户可以实现更高级的代码优化和性能调优。

# cuda-12.4的路径。
/usr/local/cuda-12.4/

# 下载cuda-samples的对应版本v12.4.1
https://github.com/NVIDIA/cuda-samples/releases/tag/v12.4.1
https://github.com/NVIDIA/cuda-samples/archive/refs/tags/v12.4.1.tar.gz

# cuda-samples的v12.4.1编译。
/Data/cuda-samples-12.4.1
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
8. 平台特定/Tegra


```




