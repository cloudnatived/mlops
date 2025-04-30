## 一.深度强化学习模型训练和推理的工具栈

深度学习框架，AI加速库，推理库

7个Python深度学习框架：PyTorch、MXNet、TensorFlow、Caffe、CNTK、Keras、PaddlePaddle。

4个大模型分布式训练加速库：DeepSpeed、Megatron-LM、Colossal-AI、BMTrain。

8个大语言模型（LLM）推理框架：vLLM、Ollama、SGLang、LMDeploy、Llama.cpp、TensorRT-LLM、HuggingFace TGI、MLC-LLM。



### 1.1. 7个Python深度学习框架

7个Python深度学习框架：PyTorch、MXNet、TensorFlow、Caffe、CNTK、Keras、PaddlePaddle

**Pytorch**

```
PyTorch是由Facebook人工智能研究小组开发的一种基于Lua编写的Torch库的Python实现的深度学习库，也是目前使用范围和体验感最好的一款深度学习框架。它的底层基于Torch，但实现与运用全部是由python来完成。该框架主要用于人工智能领域的科学研究与应用开发。PyTroch最主要的功能有两个，其一是拥有GPU张量，该张量可以通过GPU加速，达到在短时间内处理大数据的要求；其二是支持动态神经网络，可逐层对神经网络进行修改，并且神经网络具备自动求导的功能。
其优点在于：PyTorch可以使用强大的GPU加速的Tensor计算（比如：Numpy的使用）以及可以构建带有autograd的深度神经网络。PyTorch 的代码很简洁、易于使用、支持计算过程中的动态图而且内存使用很高效。
Pytorch目前主要在学术研究方向领域处于领先地位。

一些学习资料与文档：
- Awesome-pytorch-list：包含了NLP,CV,常见库，论文实现以及Pytorch的其他项目，地址：https://github.com/bharathgs/Awesome-pytorch-list
- PyTorch官方文档：官方发布文档，地址：https://pytorch.org/docs/stable/index.html
- Pytorch-handbook：pytorch hand book，地址：https://github.com/zergtant/pytorch-handbook
- PyTorch官方社区：https://discuss.pytorch.org/
```

**MXNet**

```
MXNet 是亚马逊维护的深度学习库，它拥有类似于 Theano 和 [TensorFlow](https://zhida.zhihu.com/search?content_id=238386244&content_type=Article&match_order=1&q=TensorFlow&zhida_source=entity) 的数据流图，为多 GPU 提供了良好的配置。MXNet结合了高性能、clean的代码，高级API访问和低级控制，是深度学习框架中的佼佼者。
优点：
1. 速度上有较大优势；
2. 灵活编程，支持命令式和符号式编程模型；
3. 多平台支持：可运行于多CPU、多GPU、集群、服务器、工作站甚至移动mobile phone；
4. 多语言支持：支持包括C++、Python、R、Scala、Julia、Matlab和JavaScript语言；
5. 性能优化：使用一个优化的C++后端引擎并行I/O和计算，无论使用哪种语言都能达到最佳性能；
6. 云端友好；
缺点：
1. 社区较小；
2. 入门稍微困难些；
```

**Tensorflow**

```
TensorFlow是由谷歌开源并维护，可用于各类深度学习相关的任务中。TensorFlow = Tensor + Flow，Tensor就是张量，代表N维数组；Flow即流，代表基于数据流图的计算，其特性如下：
- 支持Python、JavaScript、C ++、Java、Go、C＃、Julia和R等多种编程语言；
- GPU，iOS和Android等移动平台上都可运行；
- 入门学习难度大；
- 静态图操作；

在2017年，Tensorflow独占鳌头，处于深度学习框架的领先地位；但截至目前已经和Pytorch不争上下。Tensorflow目前主要在工业级领域处于领先地位。
```

**Caffe**

```
经典的深度学习框架，由贾扬清在加州大学伯克利分校读博期间主导开发，以C++/CUDA为主，需要编译安装，有python和matlab接口，支持单机多卡、多机多卡训练（目前已推出caffe2)，特性如下：
- 以C++/CUDA/Python代码为主，速度快，性能高；
- 代码结构清晰，可读性和可拓展性强。
- 支持命令行、Python和Matlab接口，使用方便；
- CPU和GPU之间切换方便，多GPU训练方便；
- 工具丰富，社区活跃；
- 代码修改难度较大，不支持自动求导；
- 不适合非图像（结构化）数据

官方网址：http://caffe.berkeleyvision.org/
GitHub：http://github.com/BVLC/caffe
```

**CNTK**

```
微软开源的深度学习框架，CNTK具有高度优化的内置组件，可以处理来自Python，C ++或BrainScript的多维密集或稀疏数据。能够实现CNN，DNN，RNN等任务，能够优化海量的数据集加载。

主要特性：
- CNTK性能优于其他的开源框架；
- 适合做语音任务，CNTK本就是微软语音团队开源的，自然更适合做语音任务，便于在使用RNN等模型以及时空尺度时进行卷积；

官方网址：http://cntk.ai
GitHub：http://github.com/Microsoft/CNTK
学习资料与教程：
入门介绍：[https://github.com/Microsoft/CNTK/wiki](https://link.zhihu.com/?target=https%3A//github.com/Microsoft/CNTK/wiki)
官方入门教程： [https://github.com/Microsoft/CNTK/wiki/Tutorial](https://link.zhihu.com/?target=https%3A//github.com/Microsoft/CNTK/wiki/Tutorial)
官方论坛： [https://github.com/Microsoft/CN]
```

**Keras**

```
一个小白非常友好的框架，上手简单，已被集成到Tensorflow中。Keras在高层可以调用TensorFlow、CNTK、Theano，还有更多优秀的库也在被陆续支持中。Keras的特点是能够快速搭建模型，高度模块化，搭建网络非常简洁，易于添加新模块，是高效地进行科学研究的关键。
目前Keras框架已经被集成到Tensorflow里了，在TensorFlow 2.0及其之后的版本中，Keras已经成为TensorFlow的默认高级API，使得用户可以更加方便地使用Keras构建、训练和评估深度学习模型。

1.Tensorflow更倾向于工业应用领域，适合深度学习和人工智能领域的开发者进行使用，具有强大的移植性。
2.Pytorch更倾向于科研领域，语法相对简便，利用动态图计算，开发周期通常会比Tensorflow短一些。
3.Keras因为是在Tensorflow的基础上再次封装的，所以运行速度肯定是没有Tensorflow快的；但其代码更容易理解，容易上手，用户友好性较强。

学习资料：
- https://tensorflow.rstudio.com/keras/
- https://github.com/rstudio/kera
```

**PaddlePaddle**

```
百度开源框架，支持动态图和静态图，中文文档写的非常清楚，上手比较简单；

官网链接：[https://www.paddlepaddle.org.cn/](https://link.zhihu.com/?target=https%3A//www.paddlepaddle.org.cn/)

飞桨针对不同硬件环境，提供了丰富的支持方案，生态较为完善：
1. Paddle Inference：飞桨原生推理库，用于服务器端模型部署，支持Python、C、C++、Go等语言，将模型融入业务系统的首选；
2. Paddle Serving：飞桨服务化部署框架，用于云端服务化部署，可将模型作为单独的Web服务；
3. Paddle Lite：飞桨轻量化推理引擎，用于 Mobile 及 IoT 等场景的部署，有着广泛的硬件支持；
4. Paddle.js：使用 JavaScript（Web）语言部署模型，用于在浏览器、小程序等环境快速部署模型；
5. PaddleSlim：模型压缩工具，获得更小体积的模型和更快的执行性能；
6. X2 Paddle：辅助工具，将其他框架模型转换成Paddle模型，转换格式后可以方便的使用上述5个工具；
```

参考资料：
深度学习模型训练推理框架一览！大盘点　　　　https://zhuanlan.zhihu.com/p/676152530
深度学习框架比较（Caffe, TensorFlow, MXNet, Torch, Theano）　　　　https://www.cnblogs.com/candlia/p/11920236.html



### 1.2. 4个大模型分布式训练加速库

4个大模型分布式训练加速库：DeepSpeed、Megatron-LM、Colossal-AI、BMTrain



DeepSpeed

```
微软开发，提高大模型训练效率和可扩展性:
加速训练手段:数据并行(ZeRO系列)模型并行(PP)、梯度累积、动态缩放、混合精度等,
辅助工具:分布式训练管理、内存优化和模型压缩等，帮助开发者更好管理和优化大模型训练任务。
快速迁移:通过 Python Warp 方式基于PyTorch 来构建，直接调用即完成简单迁移。
```

Megatron-LM

```
NVIDIA 开发，提高大模型分布式并行训练效率和线性度:
加速训练手段:综合数据并行(Data Parallelism)，张量并行(Tensor Parallelism)和流水线并行(Pipeline Parallelism)来复现 GPT-3 。
辅助工具:强大的数据处理 & Tokenizer支持 LLM & VLM 等基于 Transformer 结构。
```

Colossal-AI

```
Colossal-Al通过多种优化策略提高训练效率和降低显存需求:
1.加速训练手段:更加丰富张量并行策略(1D/2D/2.5D/3D-TP);
2.丰富案例:提供20+大模型DEMO和配置文件融入最新 MOE技术和 SORA;
```

BMTrain

```
BMTrain 用于训练数百亿规模参数大模型:
1.模型支持:智源研究院 Aquila 系列的模型分布式并行框架;
2.加速训练手段:支持 DeepSpeed 中并行策略深度优化;
```

参考资料：
分布式并行框架介绍 #大模型 #分布式并行 #训练    https://www.bilibili.com/video/BV1op421C7wp/
deepspeed、MegatronLM和Megatron-deepspeed的关系    https://blog.csdn.net/m0_49448331/article/details/145910339
训练大模型的九大深度学习库，哪一个最适合你？　　https://zhuanlan.zhihu.com/p/614199520
当前主流的大模型训练与推理框架的全面汇总    https://blog.csdn.net/shenhonglei1234/article/details/146136471



### 1.3. 8个大语言模型（LLM）推理框架

8个大语言模型（LLM）推理框架：vLLM、Ollama、SGLang、LMDeploy、Llama.cpp、TensorRT-LLM、HuggingFace TGI、MLC-LLM

| 平台/引擎       | 核心技术                                                 | 优势                                                   | 局限                                                | 适用场景                                 |
| --------------- | -------------------------------------------------------- | ------------------------------------------------------ | --------------------------------------------------- | ---------------------------------------- |
| vLLM            | PagedAttention、动态批处理、异步调度、多 GPU 分布式      | 高并发、低延迟，适合大规模在线服务                     | 依赖高端 GPU、代码复杂，二次开发门槛较高            | 金融、智能客服、文档处理等企业级应用     |
| ollama          | 基于 lama.cpp封装，跨平台支持、内置 1700+模型、int4 量化 | 安装便捷、易上手、低硬件要求、数据离线保障             | 并发处理能力较弱，扩展性和插件定制能力有限          | 个人原型开发、教育展示、本地隐私要求场景 |
| SGLang          | RadixAttentioN、高效缓存、结构化输出、轻量模块化架构     | 超高吞吐量、极低响应延迟、适合高并发结构化查询         | 目前仅支持力拓、对多模态任务支持能力有限            | 金融、医疗、搜索引擎等高并发实时响应场景 |
| LMDeploy        | 国产 GPU 深度适配、显存优化、多模态融合支持              | 在国产硬件上性能优异、成本优势明显，适合多模态复杂场景 | 更新迭代较慢、分布式部署和高并发处理能力待加强      | 国内企业、政府机构部署视觉语言混合任务   |
| Llama.cpp       | 纯 CPU 推理、轻量级设计、开源社区支持                    | 零硬件门槛、低成本、适合边缘和嵌入式设备               | 推理速度较慢，高并发能力有限                        | 边缘计算、特物联网、低负载场景           |
| TensorRT-LLM    | 基于 NVIDIA TenSOrRT 的深度优化、星化与预编译支持        | 极低延迟、高吞吐量、充分发挥NVIDIA GPU 优势            | 预编译过程可能带来冷启动延迟，仅限 NVIDIA CUDA 平台 | 企业级大规模在线服务、实时响应系统       |
| HuggingFace TGI | 生产级推理服务、标准化 RESTful APl、OpenAl兼容接口       | 生态成熟、稳定集成可靠、易于云端                       | 高并发定制化优化能力稍弱，部分功能依赖云端服务      | 云端部署、API推理、企业竞级生产功        |
| MLC-LLM         | 基于 Apache TVM 的编译优化、低 TTFT、实验性原型验证      | 在低并发、低延迟场景下表现突出，展示编译优化潜力       | 当前版本稳定性待提高，部署流程较复杂                | 处于研发出其、实验性应用、未大规模部署   |

参考资料：
一文详解八款主流大模型推理框架　　　　https://zhuanlan.zhihu.com/p/32055623013
大模型推理引擎vllm，sglang，transformer，exllama详细介绍和区别　　　　https://blog.csdn.net/qq_40999403/article/details/142098606
【大模型推理】大模型推理整体知识，大模型推理必看指南    https://www.bilibili.com/video/BV1dyqDYvEv2/
【AI】推理系统和推理引擎的整体架构　　　　https://blog.csdn.net/weixin_45651194/article/details/132872588


```

pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple;
mkdir -p /root/.config/pip;
cat > /root/.config/pip/pip.conf <<EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
EOF
```


