
# VLLM  SGLang  Ray TensorRT Triton dify



```
参考资料：
生产环境H200部署DeepSeek 671B 满血版全流程实战（一）：系统初始化
生产环境H200部署DeepSeek 671B 满血版全流程实战（二）：vLLM 安装详解
生产环境H200部署DeepSeek 671B 满血版全流程实战（三）：SGLang 安装详解
生产环境H200部署DeepSeek 671B 满血版全流程实战（四）：vLLM 与 SGLang 的性能大比拼
H200部署DeepSeek R1，SGLang调优性能提升2倍，每秒狂飙4000+ Tokens
猫哥手把手教你基于vllm大模型推理框架部署Qwen3-MoE
基于vLLM v1测试BFloat16 vs FP8 Qwen3-MoE模型吞吐性能的重大发现!
转载: Qwen 3 + KTransformers 0.3 (+AMX) = AI 工作站/PC

```



## 检查CUDA、NCCL、驱动，多机、多卡、容器

```
检查 CUDA 是否安装和可用：
python3 -c "import torch; print(torch.cuda.is_available())"

检查 NCCL 是否可用（Linux/GPU 环境）：
python3 -c "import torch.distributed as dist; dist.init_process_group(backend='nccl')"

多 GPU 情况运行此脚本（使用 torchrun）：
torchrun --nproc_per_node=2 test_communication.py

##################################################

PyTorch_GLOO_CPU_only.py
运行多个进程模拟多个节点，比如 2 个：
# 终端1：
python PyTorch_GLOO_CPU_only.py --backend gloo --world_size 2 --rank 0 --init_method tcp://127.0.0.1:29500
# 终端2：
python PyTorch_GLOO_CPU_only.py --backend gloo --world_size 2 --rank 1 --init_method tcp://127.0.0.1:29500

##################################################

test_communication.py

## 测试NCCL后端
#torchrun --nproc_per_node=2 test_communication.py --backend nccl
## 测试GLOO后端
#torchrun --nproc_per_node=2 test_communication.py --backend gloo
## 单GPU环境测试（跳过vLLM）
#python test_communication.py --backend nccl --skip-vllm
# 3机互连通信检查，每个node节点运行一样的命令
# NCCL_DEBUG=TRACE torchrun --nnodes 3 --nproc-per-node=8 --rdzv_backend=c10d --rdzv_endpoint=head_node_ip:8887 check_nccl.py

分布式通信测试脚本，包含了：
    动态选择 NCCL / GLOO / MPI 后端；
    NCCL（GPU）与 GLOO（CPU）通信测试；
    vLLM 的 NCCL 测试（含 CUDA Graph）。
这段代码主要用于测试 NCCL、GLOO 和 vLLM 通信后端，但它目前默认依赖 CUDA（torch.cuda 和 NCCL），这在 CPU-only 环境中无法运行。



```

## VLLM
  
  
### vLLM + DeepSeek-R1 671B 多机部署

```
NGC的PyTorch镜像。
docker pull nvcr.io/nvidia/pytorch:24.05-py3 # NGC

docker pull vllm/vllm-openai:v0.7.3 # vLLM官方镜像(推荐)


nerdctl pull nvcr.io/nvidia/tritonserver:22.01-py3
nerdctl pull vllm/vllm-openai:v0.8.5.post1
nerdctl pull nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3
nerdctl pull nvcr.io/nvidia/k8s/dcgm-exporter:4.2.0-4.1.0-ubuntu22.04

docker pull nvcr.io/nvidia/pytorch:24.05-py3


```




```
modelscope download --model 'Qwen/Qwen2-7b' --local_dir /Data/modelscope/hub/models/Qwen/Qwen2-7b
modelscope download --model 'Qwen/QwQ-32b' --local_dir /Data/modelscope/hub/models/Qwen/QwQ-32b
modelscope download --model 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B' --local_dir /Data/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
modelscope download --model 'iic/nlp_structbert_word-segmentation_chinese-base' --local_dir /Data/modelscope/hub/models/iic/nlp_structbert_word-segmentation_chinese-base

python3 -m sglang.check_env

docker run -ti --net=host --pid=host --ipc=host --privileged -v ~/.cache/huggingface:/root/.cache/huggingface --name Qwen3_0.6B  vllm/vllm-openai:v0.8.5.post1 --model Qwen/Qwen3-0.6B --tensor-parallel-size 8 --trust-remote-code --entrypoint /bin/bash

```

# SGLang
sglang的官网   https://docs.sglang.ai/backend/server_arguments.html

参考资料：
SGLang部署deepseek-ai/DeepSeek-R1-Distill-Qwen-32B实测比VLLM快30%，LMDeploy比VLLM快50% https://blog.csdn.net/weixin_46398647/article/details/145588854
```

实测使用SGLang速率为30.47 tokens/s，VLLM速率为23.23 tokens/s，快30%
LMDeploy速率为34.96 tokens/s，比VLLM快50%

SGLang启动
CUDA_VISIBLE_DEVICES=5,6,7,8 python3 -m sglang.launch_server --model ~/.cache/modelscope/hub/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/ --tp 4 --host 0.0.0.0 --port 8000

VLLM启动
CUDA_VISIBLE_DEVICES=5,6,7,8 vllm serve ~/.cache/modelscope/hub/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/ --tensor-parallel-size 4 --max-model-len 32768 --enforce-eager --served-model-name DeepSeek-R1-Distill-Qwen-32B --host 0.0.0.0

LMDeply启动
CUDA_VISIBLE_DEVICES=5,6,7,8 lmdeploy serve api_server  ~/.cache/modelscope/hub/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/ --tp 4 --server-port 8000 --model-name DeepSeek-R1-Distill-Qwen-32B

```

### Docker 安装 sglang 完整部署 DeepSeek 满血版
```
参考资料：
Docker 安装 sglang 完整部署 DeepSeek 满血版    https://zhuanlan.zhihu.com/p/29442431271

由于 DeepSeek-R1 参数量有 671B，在FP8 精度下，仅存储模型参数就需要 625GB 显存，加上KV cache 缓存及其他运行时的开销，单GPU 无法支持完全加载或高校运行，因此推荐容器分布式推理方案，更好的支持大模型的推理。

满血版定义：671B参数的deepseek不管是V3/R1，只要满足671B参数就叫满血版。
满血版划分：通常可细分为：原生满血版（FP8计算精度）、转译满血版（BF16或者FP16计算精度）、量化满血版（INT8、INT4、Q4、Q2计算精度）等版本，但是大家宣传都不会宣传XX满血版，只会宣传满血版，大家选择一定要擦亮眼睛。
原生满血版：deepseek官方支持的 FP8 混合精度，不要怀疑，官方的我们认为就是最好的
转译满血版：因为官方的 deepseek采用的是FP8混合精度，但是大部分的国产卡是不支持FP8精度的。所以要适配 deepseek，采用 BF16 或者 FP16来计算，这个方式理论上对精度影响很小，但是对计算和显存的硬件需求几乎增加一倍。

关于显存计算，如果要部署671B的官方版大模型，用FP8混合精度，最小集群显存是750GB左右；如果采用FP16或者BF16，大概需要1.4T以上。

量化满血版：很多厂家的AI卡只支持 INT8、FP16、FP32等格式，如果用FP16，单机需要1.4T显存以上，绝大多数国产AI单机没有那么大显存，为了单台机器能跑 671B deepseek，只能选择量化，量化就是通过减少计算精度，达到减少显存占用和提高吞吐效率的目的，量化后的版本会不会精准度，或者智商就变差了呢？？哈哈哈 ，大概路会吧

本文将介绍如何基于 A100 ，通过分布式推理技术(sglang)完整部署并运行 DeepSeek-R1 满血版模型。
A100 80G * 8卡的，需要4台
有文档介绍：两个 H20 节点，每个节点有 8 个 GPU，就可以部署哈
1. 环境准备
OS 选型 - Ubuntu-22.04.5
NVIDIA-Driver 安装
https://us.download.nvidia.com/tesla/535.230.02/NVIDIA-Linux-x86_64-535.230.02.run
模型下载
下载工具安装：https://www.modelscope.cn/docs/intro/quickstart

下载模型
apt install git-lfs
git lfs install
git lfs clone https://www.modelscope.cn/deepseek-ai/DeepSeek-R1.git

下载时间较长哈，建议大家从 modelscope下载模型。
modelscope download --model 'deepseek-ai/DeepSeek-R1' --local_dir '/data'

Docker 启动
#添加Docker软件包源
apt-get -y install apt-transport-https ca-certificates curl software-properties-common

curl -fsSL http://mirrors.cloud.aliyuncs.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository -y "deb [arch=$(dpkg --print-architecture)] http://mirrors.cloud.aliyuncs.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"

#安装Docker社区版本，容器运行时containerd.io，以及Docker构建和Compose插件
apt-get -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

#修改Docker存储路径
vim /etc/docker/daemon.json
{
  "data-root/graph": "/mnt/docker"  #挂载路径
}

#安装安装nvidia-container-toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get install -y nvidia-container-toolkit

#启动Docker
systemctl start docker

#设置Docker守护进程在系统启动时自动启动
sudo systemctl enable docker

分布式推理
本次使用的是分布式推理框架是 sglang：https://github.com/sgl-project/sglang
Docs：https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3
DeepSeek 模型格式转换（FP8->BF16）
模型格式转换，因为 A100 机型不支持 FP8 类型的数据格式，所以第一步必须将 DeepSeek-R1 的模型权重转换为 BF16 格式。
格式转换脚本：https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/fp8_cast_bf16.py

主要通过weight_dequant函数将FP8权重转换为BF16格式，参考文档：
快来学习一下：高性能 ：DeepSeek-V3 inference 推理时反量化实现 fp8_cast_bf16

copy 转换后的模型权重至另外的 3 台 A100 机器。

RDMA 网络配置
分布式推理需要用到 RDMA 网络进行网络通信，否则推理效率非常低下。
实测：在未使用 RDMA 网络时，DeepSeek-R1 在8h之内没有加载完成。

检查 mellanox 网卡硬件设备是否存在：lspci | grep -i mellanox
RDMA 驱动安装：https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/
参考文档：https://sulao.cn/post/977.html

启动 RDMA 相关服务。
检测 RDMA 设备：ofed_info
启动 sglang 分布式推理

补充说明
启动命令时，先启动主节点，后启动副节点
所有节点的 --dist-init-addr 均需设置为主节点IP
在服务启动后，发起请求时需指定为主节点IP

# node01
docker run -d \
     --privileged \
     --gpus all \
     -e NCCL_SOCKET_IFNAME=eth0 \
     -e NCCL_NVLS_ENABLE=0 \
     -e NCCL_DEBUG=INFO \
     -e NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4 \
     -e NCCL_IB_QPS_PER_CONNECTION=8 \
     -e NCCL_SOCKET_FAMILY=AF_INET \
     -e NCCL_IB_TIMEOUT=22 \
     -e NCCL_IB_DISABLE=0 \
     -e NCCL_IB_RETRY_CNT=12 \
     -e NCCL_IB_GID_INDEX=3 \
     --network host \
     --shm-size 32g \
     -p 30000:30000 \
     -v /data/deepseek-R1-fp16:/data/deepseek-R1-fp16 \
     --ipc=host \
     <registry.io>/lmsysorg/sglang:latest \
     python3 -m sglang.launch_server --model-path /data/deepseek-R1-fp16/ --tp 32 --dist-init-addr 192.168.5.4:5000 --nnodes 4 --node-rank 0 --trust-remote-code --host 0.0.0.0 --port 30000
     
# node02
docker run -d \
     --privileged \
     --gpus all \
     -e NCCL_SOCKET_IFNAME=eth0 \
     -e NCCL_NVLS_ENABLE=0 \
     -e NCCL_DEBUG=INFO \
     -e NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4 \
     -e NCCL_IB_QPS_PER_CONNECTION=8 \
     -e NCCL_SOCKET_FAMILY=AF_INET \
     -e NCCL_IB_TIMEOUT=22 \
     -e NCCL_IB_DISABLE=0 \
     -e NCCL_IB_RETRY_CNT=12 \
     -e NCCL_IB_GID_INDEX=3 \
     --network host \
     --shm-size 32g \
     -p 30000:30000 \
     -v /data/deepseek-R1-fp16:/data/deepseek-R1-fp16 \
     --ipc=host \
     <registry.io>/lmsysorg/sglang:latest \
     python3 -m sglang.launch_server --model-path /data/deepseek-R1-fp16/ --tp 32 --dist-init-addr 192.168.5.4:5000 --nnodes 4 --node-rank 1 --trust-remote-code     <registry.io>/sre/lmsysorg/sglang:latest \


节点 02 ，节点 03 以此忘后推
web 页面
open webui进行支持
https://github.com/open-webui/open-webui?tab=readme-ov-file#installation-with-default-configuration
#拉取Open WebUI镜像。
sudo docker pull alibaba-cloud-linux-3-registry.cn-hangzhou.cr.aliyuncs.com/alinux3/python:3.11.1

#启动Open WebUI服务。

#设置模型服务地址
OPENAI_API_BASE_URL=http://127.0.0.1:30000/v1

# 创建数据目录,确保数据目录存在并位于/mnt下
sudo mkdir -p /mnt/open-webui-data

#启动open-webui服务 
#需注意系统盘空间，建议100GB以上
sudo docker run -d -t --network=host --name open-webui \
-e ENABLE_OLLAMA_API=False \
-e OPENAI_API_BASE_URL=${OPENAI_API_BASE_URL} \
-e DATA_DIR=/mnt/open-webui-data \
-e HF_HUB_OFFLINE=1 \
-v /mnt/open-webui-data:/mnt/open-webui-data \
alibaba-cloud-linux-3-registry.cn-hangzhou.cr.aliyuncs.com/alinux3/python:3.11.1 \
/bin/bash -c "pip config set global.index-url http://mirrors.cloud.aliyuncs.com/pypi/simple/ && \
pip config set install.trusted-host mirrors.cloud.aliyuncs.com && \
pip install --upgrade pip && \
pip install open-webui && \
mkdir -p /usr/local/lib/python3.11/site-packages/google/colab && \
open-webui serve"

#行以下命令，实时监控下载进度，等待下载结束。
docker logs -f open-webui

#在日志输出中寻找类似以下的消息：
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
#表示服务已经成功启动并在端口8080上监听。

#本地物理机上使用浏览器访问http://<公网IP地址>:8080，首次登录时，请根据提示创建管理员账号。
```



### SGLang部署DeepSeek-70B大模型

```
参考资料：
手把手教你用SGLang部署DeepSeek-70B大模型（附避坑指南）  https://blog.csdn.net/2501_91377542/article/details/147441180

一、环境准备
1.1 硬件配置建议
显卡：至少需要8张A100 80G
内存：建议128G以上
硬盘：预留500G空间
如果显存不足，可以试试量化版本（比如4bit量化）

1.2 软件依赖安装
# 安装万能依赖包
sudo apt-get update && sudo apt-get install -y git curl wget python3-pip

二、模型下载（两种方式任选）
2.1 官方推荐方法（ModelScope）
# 安装工具包
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

# 创建模型仓库
mkdir -p /data/deepseek-ai/models/deepseek-70b

# 开下！(记得连好VPN)
modelscope download --local_dir /data/deepseek-ai/models/deepseek-70b \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B

2.2 HuggingFace备份方案
# 安装huggingface-cli
pip install huggingface_hub

# 下载模型（需要Access Token）
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
    --local-dir /data/deepseek-ai/models/deepseek-70b

三、Docker环境配置
3.1 安装Docker全家桶
# 一键安装脚本
curl -fsSL https://get.docker.com | bash -s docker

# 配置镜像加速（解决下载慢问题）
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": [
    "https://docker.211678.top",
    "https://docker.m.daocloud.io"
  ]
}
EOF

# 重启服务
sudo systemctl restart docker

3.2 验证GPU支持
# 运行测试容器 docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi #看到显卡信息就成功啦！

四、SGLang服务部署
4.1 拉取最新镜像
docker pull lmsysorg/sglang:latest

4.2 单命令启动（适合快速测试）
docker run -itd --name sglang_ds70 \
    --gpus all --ipc=host --shm-size=16g \
    -v /data/deepseek-ai:/data \
    --network=host lmsysorg/sglang:latest \
    python3 -m sglang.launch_server \
    --model "/data/models/deepseek-70b" \
    --tp 8 --mem-fraction-static 0.8 \
    --trust-remote-code --dtype bfloat16 \
    --host 0.0.0.0 --port 30000 \
    --api-key token-abc123 \
    --served-model-name DeepSeek-70B

4.3 生产级部署（推荐docker-compose）
# docker-compose.yml
version: '3.9'
services:
  sglang:
    image: lmsysorg/sglang:latest
    volumes:
      - /data/deepseek-ai:/data
    ports:
      - "30000:30000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 8
              capabilities: [gpu]
    command: --model-path /data/models/deepseek-70b
             --tp 8 --port 30000
             --api-key my-secret-token
docker-compose up -d

五、服务验证与测试
5.1 健康检查
curl http://localhost:30000/health
# 返回{"status":"healthy"}就成功啦！

5.2 API调用示例
import requests

url = "http://localhost:30000/v1/chat/completions"
headers = {
    "Authorization": "Bearer my-secret-token",
    "Content-Type": "application/json"
}

data = {
    "model": "DeepSeek-70B",
    "messages": [
        {"role": "user", "content": "用鲁迅的风格写一篇关于秋天的散文"}
    ],
    "temperature": 0.7,
    "max_tokens": 500
}

response = requests.post(url, json=data, headers=headers)
print(response.json()['choices'][0]['message']['content'])

六、常见问题排雷指南
报错：CUDA out of memory
解决方案：
减少--tp参数值（比如从8改为4）
使用--dtype float16代替bfloat16
添加--quantization awq启用4bit量化

模型下载中断
解决方案：
# 续传下载（ModelScope专用）
modelscope download --local-dir /data/deepseek-ai/models/deepseek-70b \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
    --resume-download

七、性能优化小技巧
开启FlashAttention：添加--flash-attn参数
调整批处理大小：设置--max-num-batched-tokens 4096
使用vLLM后端：添加--backend vllm参数
监控GPU状态：watch -n 1 nvidia-smi

```


Sglang部署大模型常用参数详解    


## TensorRT Triton

```
TensorRT&Triton学习笔记(一)：triton和模型部署+client https://zhuanlan.zhihu.com/p/482170985
TensorRT详细入门指北，如果你还不了解TensorRT，过来看看吧  https://blog.csdn.net/IAMoldpan/article/details/117908232
Triton + TensorRT 推理模型部署  https://blog.csdn.net/weixin_39403185/article/details/147105599
深度学习模型部署 - Triton 篇 https://juejin.cn/post/7221444501956067388
AI模型部署：Triton Inference Server模型部署框架简介和快速实践 https://www.jianshu.com/p/a7fc654f678c

Triton Inference Server简介
Triton Inference Server是一款开源的推理服务框架，它的核心库基于C++编写的，旨在在生产环境中提供快速且可扩展的AI推理能力，具有以下优势

支持多种深度学习框架：包括PyTorch，Tensorflow，TensorRT，ONNX，OpenVINO等产出的模型文件
至此多种机器学习框架：支持对树模型的部署，包括XGBoost，LightGBM等
支持多种推理协议：支持HTTP，GRPC推理协议
服务端支持模型前后处理：提供后端API，支持将数据的前处理和模型推理的后处理在服务端实现
支持模型并发推理：支持多个模型或者同一模型的多个实例在同一系统上并行执行
支持动态批处理（Dynamic batching）：支持将一个或多个推理请求合并成一个批次，以最大化吞吐量
支持多模型的集成流水线：支持将多个模型进行连接组合，将其视作一个整体进行调度管理
Triton Inference Server架构如下图所示，从客户端请求开始，到模型调度处理，模型仓库管理和推理，响应返回，服务状态监控等。




CPU版本的启动：
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:22.01-py3 tritonserver --model-repository=/models
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3 tritonserver --model-repository=/models


GPU版本的启动，使用1个gpu：
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:22.01-py3 tritonserver --model-repository=/models
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3 tritonserver --model-repository=/models




```



## dify
```
git clone https://github.com/langgenius/dify.git  
cd dify/docker 
copy .env.example .env;

cat >> .env<<EOF
CONSOLE_URL=http://localhost
SERVICE_API_URL=http://localhost
EOF

sed -i 's#UPLOAD_FILE_SIZE_LIMIT=15#UPLOAD_FILE_SIZE_LIMIT=1500#' .env;
sed -i 's#UPLOAD_FILE_BATCH_LIMIT=5#UPLOAD_FILE_BATCH_LIMIT=500#' .env;
sed -i 's#UPLOAD_IMAGE_FILE_SIZE_LIMIT=10#UPLOAD_IMAGE_FILE_SIZE_LIMIT=1000#' .env;
sed -i 's#UPLOAD_VIDEO_FILE_SIZE_LIMIT=100#UPLOAD_VIDEO_FILE_SIZE_LIMIT=10000#' .env;
sed -i 's#UPLOAD_AUDIO_FILE_SIZE_LIMIT=50#UPLOAD_AUDIO_FILE_SIZE_LIMIT=5000#' .env;
sed -i 's#WORKFLOW_FILE_UPLOAD_LIMIT=10#WORKFLOW_FILE_UPLOAD_LIMIT=1000#' .env;

nerdctl compose up -d  
nerdctl compose restart 


# bge-m3 nomic-embed-text
ollama pull bge-m3

# ollama
pip3 install ollama
curl -fsSL https://ollama.com/install.sh | sh

# /etc/systemd/system/ollama.service
Environment="OLLAMA_HOST=0.0.0.0:11434"
ExecStart=/usr/local/bin/ollama serve


# ollama in docker
nerdctl pull ollama/ollama:latest
nerdctl pull dhub.kubesre.xyz/ollama/ollama:latest

root@250:/Data/DIFI# nerdctl image inspect dhub.kubesre.xyz/ollama/ollama:latest | grep -i version
        "DockerVersion": "",
                "org.opencontainers.image.version": "20.04"


#nerdctl run -d --gpus=all --restart=always -v /root/project/docker/ollama:/root/project/.ollama -p 11434:11434 --name ollama ollama/ollama
# gpu机器上。
nerdctl run -d --gpus=all --restart=always -v /root/project/docker/ollama:/root/project/.ollama -p 11434:11434 --name ollama ollama/ollama:20.04

# cpu机器上。
nerdctl run -d --restart=always -v /root/project/docker/ollama:/root/project/.ollama -p 11434:11434 --name ollama ollama/ollama:20.04
docker run -d --restart=always -v /root/project/docker/ollama:/root/project/.ollama -p 11434:11434 --name ollama ollama/ollama:20.04


# 进入ollama容器
nerdctl exec -it f9d215c9b0a0 /bin/bash

bge-m3 
ollama pull bge-m3

ollama pull deepseek-r1:7b
ollama pull deepseek-r1:1.5b

# 跑起来那个windows11
nerdctl run 25dc03aa3976

http://10.22.0.2:11434/
http://10.0.10.250:11434



set OLLAMA_HOST=0.0.0.0:11434
ollama serve

OLLAMA_HOST

/root/datapipe 10.0.10.248 80 10.0.10.250 80
/root/datapipe 10.0.10.250 11434 10.22.0.22 11434

这里是解决weaviate的启动。
https://blog.csdn.net/xmbpc/article/details/144467990
https://cloud.tencent.com/developer/news/1552212
https://docs.dify.ai/zh-hans/getting-started/install-self-hosted/local-source-code
https://docs.dify.ai/zh-hans/getting-started/install-self-hosted/docker-compose
https://z0yrmerhgi8.feishu.cn/wiki/JYsNwWXZpiZzJYkem57cOiFPnhQ
https://blog.csdn.net/ssw_1990/article/details/140258162
https://z0yrmerhgi8.feishu.cn/wiki/PLLHwT9iDiLI5xkKKThczUbtnqe

https://zhuanlan.zhihu.com/p/691190576


cd ../docker
cp middleware.env.example middleware.env
nerdctl compose -f docker-compose.middleware.yaml --profile weaviate -p dify up -d


387b38da07b2    docker.io/library/postgres:15-alpine          "docker-entrypoint.s…"    6 seconds ago     Up        0.0.0.0:5432->5432/tcp                            dify-db-1
27a78410ed9e    docker.io/ubuntu/squid:latest                 "sh -c cp /docker-en…"    8 seconds ago     Up        0.0.0.0:3128->3128/tcp, 0.0.0.0:8194->8194/tcp    dify-ssrf_proxy-1
f4350a4c46a6    docker.io/langgenius/dify-sandbox:0.2.10      "/main"                   11 seconds ago    Up                                                          dify-sandbox-1
6a4c592f3123    docker.io/semitechnologies/weaviate:1.19.0    "/bin/weaviate --hos…"    13 seconds ago    Up        0.0.0.0:8080->8080/tcp                            dify-weaviate-1
8bc3cd16cfc9    docker.io/library/redis:6-alpine              "docker-entrypoint.s…"    14 seconds ago    Up        0.0.0.0:6379->6379/tcp                            dify-redis-1
ad227a5d4c03    docker.io/ubuntu/squid:latest                 "sh -c cp /docker-en…"    2 minutes ago     Up                                                          docker-ssrf_proxy-1
19663adf3dae    docker.io/library/nginx:latest                "sh -c cp /docker-en…"    3 minutes ago     Up        0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp          docker-nginx-1
2dd3f27e8977    docker.io/langgenius/dify-api:0.15.3          "/bin/bash /entrypoi…"    3 minutes ago     Up                                                          docker-api-1
1b200be89b3a    docker.io/langgenius/dify-sandbox:0.2.10      "/main"                   3 minutes ago     Up                                                          docker-sandbox-1
1abf4a5a0ba0    docker.io/langgenius/dify-api:0.15.3          "/bin/bash /entrypoi…"    3 minutes ago     Up                                                          docker-worker-1
8fe0f76b2ba9    docker.io/library/redis:6-alpine              "docker-entrypoint.s…"    3 minutes ago     Up                                                          docker-redis-1
c1fe7524a0ec    docker.io/langgenius/dify-web:0.15.3          "/bin/sh ./entrypoin…"    3 minutes ago     Up                                                          docker-web-1
107b83060ab2    docker.io/library/postgres:15-alpine          "docker-entrypoint.s…"    3 minutes ago     Up                                                          docker-db-1
f9d215c9b0a0    docker.io/ollama/ollama:20.04                 "/bin/ollama serve"       39 hours ago      Up        0.0.0.0:11434->11434/tcp                          ollama



nerdctl compose -f docker-compose.yaml up -d
8e8a0370c633    docker.io/ubuntu/squid:latest               "sh -c cp /docker-en…"    19 seconds ago    Up                                                    docker-ssrf_proxy-1
9273cae5bb0a    docker.io/langgenius/dify-api:0.15.3        "/bin/bash /entrypoi…"    21 seconds ago    Up                                                    docker-worker-1
fc1b61e29c27    docker.io/langgenius/dify-sandbox:0.2.10    "/main"                   22 seconds ago    Up                                                    docker-sandbox-1
432fe41bf203    docker.io/library/nginx:latest              "sh -c cp /docker-en…"    23 seconds ago    Up        0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp    docker-nginx-1
9b0c7a1b3483    docker.io/langgenius/dify-api:0.15.3        "/bin/bash /entrypoi…"    24 seconds ago    Up                                                    docker-api-1
aee5d5aa0172    docker.io/library/redis:6-alpine            "docker-entrypoint.s…"    25 seconds ago    Up                                                    docker-redis-1
7d8c9e26222c    docker.io/library/postgres:15-alpine        "docker-entrypoint.s…"    26 seconds ago    Up                                                    docker-db-1
31917c52173b    docker.io/langgenius/dify-web:0.15.3        "/bin/sh ./entrypoin…"    27 seconds ago    Up                                                    docker-web-1
f9d215c9b0a0    docker.io/ollama/ollama:20.04               "/bin/ollama serve"       39 hours ago      Up        0.0.0.0:11434->11434/tcp                    ollama



```

## docker-ce
```
https://linux.cn/article-14871-1.html  #用这个文档安装docker-ce

apt -y install apt-transport-https ca-certificates curl software-properties-common gnupg lsb-release;
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg;
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null;
apt -y update;


#选择版本
apt-cache madison docker-ce
apt install docker-ce=5:20.10.16~3-0~ubuntu-jammy docker-ce-cli=5:20.10.16~3-0~ubuntu-jammy containerd.io

# 直接安装。
apt -y install docker-ce docker-ce-cli containerd.io docker-compose-plugin

cat <<-EOF > /etc/docker/daemon.json
{
  "registry-mirrors": [
        "https://docker.1ms.run",
        "https://docker.xuanyuan.me"
        ]
}
EOF
systemctl daemon-reload
systemctl restart docker

#测试docker-ce
docker run hello-world


###########################
  weaviate:
    image: semitechnologies/weaviate:1.19.0
    ports:
      - "8000:8000"
    profiles:
      - ''
      - weaviate
    restart: always
    volumes:
      # Mount the Weaviate data directory to the con tainer.
      - ./volumes/weaviate:/var/lib/weaviate
    environment:

docker compose -f docker-compose.yaml up -d;
docker compose -f docker-compose.yaml up -d;
###########################


```


## Windows in Docker Container
```
windowhttps://zhuanlan.zhihu.com/p/686351917  # 把 Windows 装进 Docker 容器里

git clone https://github.com/dockur/windows.git
cd windows
nerdctl build -t dockurr/windows .
#需要启动 buildkit.service

docker compose up  
nerdctl compose up

------------------------------------ cat compose.yml
version: "3"
services:
  windows:
    image: dockurr/windows
    container_name: windows
    environment:
      VERSION: "win11"
    devices:
      - /dev/kvm
    cap_add:
      - NET_ADMIN
    ports:
      - 8006:8006
      - 3389:3389/tcp
      - 3389:3389/udp
    stop_grace_period: 2m
    restart: on-failure
------------------------------------


nerdctl container ls -a |grep windows


Value	Version	Size
11	Windows 11 Pro	5.4 GB
11l	Windows 11 LTSC	4.7 GB
11e	Windows 11 Enterprise	4.0 GB
10	Windows 10 Pro	5.7 GB
10l	Windows 10 LTSC	4.6 GB
10e	Windows 10 Enterprise	5.2 GB
8e	Windows 8.1 Enterprise	3.7 GB
7e	Windows 7 Enterprise	3.0 GB
ve	Windows Vista Enterprise	3.0 GB
xp	Windows XP Professional	0.6 GB

compose.yml
services:
  windows:
    image: dockurr/windows
    container_name: windows-10 # windows10
    environment:
      VERSION: "10"   # windows10
    devices:
      - /dev/kvm
      - /dev/net/tun
    cap_add:
      - NET_ADMIN
    ports:
      - 8006:8006
      - 3389:3389/tcp
      - 3389:3389/udp
    restart: always
    stop_grace_period: 2m

root@250:/opt/windows-10# cat Dockerfile 
ARG VERSION_ARG="latest"
FROM scratch AS build-amd64

COPY --from=qemux/qemu:6.20 / /

ARG DEBCONF_NOWARNINGS="yes"
ARG DEBIAN_FRONTEND="noninteractive"
ARG DEBCONF_NONINTERACTIVE_SEEN="true"

RUN set -eu && \
    apt-get update && \
    apt-get --no-install-recommends -y install \
        bc \
        jq \
        curl \
        7zip \
        wsdd \
        samba \
        xz-utils \
        wimtools \
        dos2unix \
        cabextract \
        genisoimage \
        libxml2-utils \
        libarchive-tools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --chmod=755 ./src /run/
COPY --chmod=755 ./assets /run/assets

ADD --chmod=664 https://github.com/qemus/virtiso-whql/releases/download/v1.9.45-0/virtio-win-1.9.45.tar.xz /drivers.txz

FROM dockurr/windows-arm:${VERSION_ARG} AS build-arm64
FROM build-${TARGETARCH}

ARG VERSION_ARG="0.00"
RUN echo "$VERSION_ARG" > /run/version

VOLUME /storage
EXPOSE 8006 3389

ENV VERSION="10"    # windows10
ENV RAM_SIZE="32G"
ENV CPU_CORES="6"
ENV DISK_SIZE="64G"

ENTRYPOINT ["/usr/bin/tini", "-s", "/run/entry.sh"]
```



