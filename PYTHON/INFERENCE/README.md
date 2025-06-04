
# VLLM  SGLang  Ray 



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

# VLLM
  
  
## vLLM + DeepSeek-R1 671B 多机部署

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

参考资料：
Docker 安装 sglang 完整部署 DeepSeek 满血版    https://zhuanlan.zhihu.com/p/29442431271
```
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

参考资料：
手把手教你用SGLang部署DeepSeek-70B大模型（附避坑指南）  https://blog.csdn.net/2501_91377542/article/details/147441180
```

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
```
常用启动命令
要启用多GPU张量并行性，请添加 --tp 2。如果报告错误“这些设备之间不支持对等访问”，请在服务器启动命令中添加 --enable-p2p-check。
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 2

要启用多 GPU 数据并行，请添加–dp 2。如果内存足够，数据并行对吞吐量更有利。它也可以与张量并行一起使用。以下命令总共使用 4 个 GPU。我们建议使用SGLang Router进行数据并行。
python -m sglang_router.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --dp 2 --tp 2

如果在服务过程中出现内存不足错误，请尝试通过设置较小的值来减少 KV 缓存池的内存使用量–mem-fraction-static。默认值为0.9。
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --mem-fraction-static 0.7

如果在长提示的预填充过程中看到内存不足错误，请尝试设置较小的分块预填充大小。
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --chunked-prefill-size 4096

要启用torch.compile加速，请添加 --enable-torch-compile。它可以在小批量大小上加速小型模型。但目前这不适用于FP8。你可以参考“为torch.compile启用缓存”以获取更多详情。
要启用torchao量化，请添加 --torchao-config int4wo-128。它也支持其他的量化策略（INT8/FP8）。
要启用fp8权重量化，在fp16检查点上添加 --quantization fp8 或直接加载一个fp8检查点，无需指定任何参数。
要启用fp8 kv缓存量化，请添加 --kv-cache-dtype fp8_e5m2。
如果模型在Hugging Face的tokenizer中没有聊天模板，可以指定一个自定义聊天模板。
要在多个节点上运行张量并行，请添加 --nnodes 2。如果你有两个节点，每个节点上有两个GPU，并希望运行TP=4，假设sgl-dev-0是第一个节点的主机名且50000是一个可用端口，你可以使用以下命令。如果遇到死锁，请尝试添加 --disable-cuda-graph。
# Node 0
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 4 --dist-init-addr sgl-dev-0:50000 --nnodes 2 --node-rank 0

# Node 1
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 4 --dist-init-addr sgl-dev-0:50000 --nnodes 2 --node-rank 1

模型和分词器参数说明：
model_path: 模型存放的路径，该模型将会被加载用于服务。
tokenizer_path: 默认与model_path相同。这是分词器文件所在的路径。
tokenizer_mode: 默认为auto模式，具体不同模式可以参考相关文档。
load_format: 权重文件的格式，默认是*.safetensors/*.bin。
trust_remote_code: 如果设置为True，则使用本地缓存的配置文件；否则使用HuggingFace中的远程配置。
dtype: 用于模型的数据类型，默认是bfloat16。
kv_cache_dtype: kv缓存使用的数据类型，默认与dtype相同。
context_length: 模型能够处理的token数量，包括输入的tokens。请注意扩展默认值可能会导致奇怪的行为。
device: 模型部署的设备，默认是cuda。
chat_template: 使用的聊天模板。不使用默认模板可能导致意外的回复。对于多模态聊天模板，请参阅相关部分。确保传递正确的chat_template，否则可能导致性能下降。
is_embedding: 设置为true以执行嵌入/编码和奖励任务。
revision: 如果需要使用模型的特定版本，可以通过此参数调整。
skip_tokenizer_init: 设置为true时，提供tokens给引擎并直接获取输出tokens，通常在RLHF中使用。请参考提供的示例。
json_model_override_args: 使用提供的JSON覆盖模型配置。
delete_ckpt_after_loading: 加载模型后删除模型检查点。

服务：HTTP & API
HTTP服务器配置
port 和 host: 设置HTTP服务器的主机地址。默认情况下，host: str = "127.0.0.1"（即本地回环地址）和port: int = 30000。

API配置
api_key: 设置服务器和兼容OpenAI的API的API密钥。
file_storage_path: 用于存储从API调用中上传或生成的文件的目录。
enable_cache_report: 如果设置了此选项，则在响应使用情况中包括缓存token使用的详细信息。

并行处理
张量并行
tp_size: 模型权重分片所用的GPU数量。主要用于节省内存而不是提高吞吐量，详情见相关博客文章。

数据并行
dp_size: 将被弃用。模型的数据并行副本的数量。推荐使用SGLang路由器代替当前的简单数据并行。
load_balance_method: 将被弃用。数据并行请求的负载均衡策略。

专家并行
enable_ep_moe: 启用专家并行，将MoE模型中的专家分布到多个GPU上。
ep_size: EP（专家并行）的大小。请以tp_size=ep_size的方式分割模型权重，具体基准测试参见PR。如果未设置，ep_size会自动设置为tp_size。

内存和调度
mem_fraction_static: 用于静态内存（如模型权重和KV缓存）的空闲GPU内存的比例。如果构建KV缓存失败，应该增加此值；如果CUDA内存不足，则应减少。
max_running_requests: 并发运行的最大请求数量。
max_total_tokens: 可以存储到KV缓存中的最大token数。主要用于调试。
chunked_prefill_size: 以这些大小的块执行预填充。较大的块大小加快了预填充阶段但增加了VRAM消耗。如果CUDA内存不足，应减少此值。
max_prefill_tokens: 一个预填充批次中接受的token预算。实际数字是此参数与context_length之间的最大值。
schedule_policy: 控制单个引擎中等待预填充请求的处理顺序的调度策略。
schedule_conservativeness: 用于调整服务器在接收新请求时的保守程度。高度保守的行为会导致饥饿，而较低的保守性会导致性能下降。
cpu_offload_gb: 为卸载到CPU的模型参数保留的RAM量（GB）。

其他运行时选项
stream_interval: 流式响应的间隔（按token计）。较小的值使流式传输更平滑，较大的值提供更好的吞吐量。
random_seed: 用于强制更确定性的行为。
watchdog_timeout: 调整看门狗线程的超时时间，在批处理生成花费过长时间时终止服务器。
download_dir: 用于覆盖Hugging Face默认的模型权重缓存目录。
base_gpu_id: 用于调整第一个用于跨可用GPU分配模型的GPU。
allow_auto_truncate: 自动截断超过最大输入长度的请求。

日志记录
log_level: 全局日志详细级别。
log_level_http: HTTP服务器日志的独立详细级别（如果未设置，默认为log_level）。
log_requests: 记录所有请求的输入和输出以进行调试。
show_time_cost: 打印或记录内部操作的详细计时信息（有助于性能调优）。
enable_metrics: 导出类似于Prometheus的请求使用情况和性能指标。
decode_log_interval: 记录解码进度的频率（按token计）。

多节点分布式服务
dist_init_addr: 用于初始化PyTorch分布式后端的TCP地址（例如192.168.0.2:25000）。
nnodes: 集群中的总节点数。参考如何运行Llama 405B模型。
node_rank: 在分布式设置中该节点在nnodes中的排名（ID）。
LoRA
lora_paths: 可以为您的模型提供一系列适配器作为列表。每个批次元素都会获得应用相应LoRA适配器的模型响应。目前cuda_graph和radix_attention不支持此选项，因此需要手动禁用。
max_loras_per_batch: 运行批次中包括基本模型在内的最大LoRAs数量。
lora_backend: LoRA模块运行GEMM内核的后端，可以是triton或flashinfer之一，默认为triton。

内核后端
attention_backend: 注意力计算和KV缓存管理的后端。
sampling_backend: 采样的后端。

约束解码
grammar_backend: 约束解码的语法后端。详细使用方法见相关文档。
constrained_json_whitespace_pattern: 与Outlines语法后端一起使用，允许JSON包含语法上的换行符、制表符或多空格。详情见此处。

推测解码
speculative_draft_model_path: 用于推测解码的草稿模型路径。
speculative_algorithm: 推测解码的算法。当前仅支持Eagle。注意，在使用eagle推测解码时，radix缓存、分块预填充和重叠调度器将被禁用。
speculative_num_steps: 在验证前运行多少次草稿。
speculative_num_draft_tokens: 草稿中提议的token数量。
speculative_eagle_topk: 每一步为Eagle保留进行验证的顶级候选者数量。
speculative_token_map: 可选，指向FR-Spec高频token列表的路径，用于加速Eagle。

双稀疏性
enable_double_sparsity: 启用双稀疏性，提高吞吐量。
ds_channel_config_path: 双稀疏配置。关于如何为您的模型生成配置，请参阅此仓库。
ds_heavy_channel_num: 每层要保持的通道索引数量。
ds_heavy_token_num: 解码期间用于注意力的token数量。如果批次中的min_seq_len小于该数字，则跳过稀疏解码。
ds_heavy_channel_type: 重型通道的类型。可以是q、k或qk。
ds_sparse_decode_threshold: 如果批次中的max_seq_len小于该阈值，则不应用稀疏解码。

调试选项
disable_radix_cache: 禁用Radix后端用于前缀缓存。
disable_cuda_graph: 禁用cuda图用于模型前向传播。如果遇到无法纠正的CUDA ECC错误，请使用此选项。
disable_cuda_graph_padding: 当需要填充时禁用cuda图。在其他情况下仍然使用cuda图。
disable_outlines_disk_cache: 禁用outlines语法后端的磁盘缓存。
disable_custom_all_reduce: 禁用自定义all reduce内核的使用。
disable_mla: 禁用Deepseek模型的多头潜在注意力(MLA)。
disable_overlap_schedule: 禁用重叠调度器。
enable_nan_detection: 开启此选项会使采样器在logits包含NaN时打印警告。
enable_p2p_check: 关闭默认允许始终进行GPU访问时的p2p检查。
triton_attention_reduce_in_fp32: 在triton内核中，这会将中间注意力结果转换为float32。

优化选项
enable_mixed_chunk: 启用混合预填充和解码，详见讨论。
enable_dp_attention: 启用Deepseek模型的数据并行注意力。请注意，您需要选择dp_size = tp_size。
enable_torch_compile: 使用torch编译模型。注意，编译模型耗时较长但能显著提升性能。编译后的模型也可以缓存以备将来使用。
torch_compile_max_bs: 使用torch_compile时的最大批量大小。
cuda_graph_max_bs: 使用cuda图时调整最大批量大小。默认根据GPU规格为您选择。
cuda_graph_bs: CudaGraphRunner捕获的批量大小。默认自动完成。
torchao_config: 实验性功能，使用torchao优化模型。可能的选择有：int8dq, int8wo, int4wo-<group_size>, fp8wo, fp8dq-per_tensor, fp8dq-per_row。
triton_attention_num_kv_splits: 用于调整triton内核中的KV分割数量。默认是8。
enable_flashinfer_mla: 使用带有flashinfer MLA包装器的注意力后端用于Deepseek模型。提供此参数时，将覆盖attention_backend参数。
flashinfer_mla_disable_ragged: 当启用enable_flashinfer_mla时，应使用此选项禁用ragged预填充包装器。


参数概览
-h, --help            显示帮助信息并退出
  --model-path MODEL_PATH
                        模型权重的路径。可以是本地文件夹或Hugging Face仓库ID。
  --tokenizer-path TOKENIZER_PATH
                        分词器的路径。
  --host HOST           服务器的主机地址。
  --port PORT           服务器的端口。
  --tokenizer-mode {auto,slow}
                        分词器模式。'auto'会使用可用的快速分词器，而'slow'总是使用慢速分词器。
  --skip-tokenizer-init
                        如果设置，跳过初始化分词器，并在生成请求时传递input_ids。
  --load-format {auto,pt,safetensors,npcache,dummy,gguf,bitsandbytes,layered}
                        要加载的模型权重格式。“auto”将尝试以safetensors格式加载权重，如果不可用则回退到pytorch bin格式。“pt”将以pytorch bin格式加载权重。“safetensors”将以safetensors格式加载权重。“npcache”将以pytorch格式加载权重并在numpy缓存中存储以加快加载速度。“dummy”将使用随机值初始化权重，主要用于性能分析。“gguf”将以gguf格式加载权重。“bitsandbytes”将使用bitsandbytes量化加载权重。“layered”逐层加载权重，以便在一个层被量化之前加载另一个层，从而减小峰值内存占用。
  --trust-remote-code   是否允许Hub上自定义模型在其自己的建模文件中定义。
  --dtype {auto,half,float16,bfloat16,float,float32}
                        模型权重和激活的数据类型。* "auto"对FP32和FP16模型使用FP16精度，对BF16模型使用BF16精度。 * "half"为FP16。推荐用于AWQ量化。 * "float16"与"half"相同。 * "bfloat16"在精度和范围之间取得平衡。 * "float"是FP32精度的简写。 * "float32"为FP32精度。
  --kv-cache-dtype {auto,fp8_e5m2,fp8_e4m3}
                        KV缓存存储的数据类型。“auto”将使用模型数据类型。“fp8_e5m2”和“fp8_e4m3”支持CUDA 11.8+。
  --quantization-param-path QUANTIZATION_PARAM_PATH
                        包含KV缓存缩放因子的JSON文件的路径。当KV缓存数据类型为FP8时通常需要提供。否则，默认缩放因子为1.0，可能导致准确性问题。
  --quantization {awq,fp8,gptq,marlin,gptq_marlin,awq_marlin,bitsandbytes,gguf,modelopt,w8a8_int8}
                        量化方法。
  --context-length CONTEXT_LENGTH
                        模型的最大上下文长度。默认为None（将使用模型config.json中的值）。
  --device {cuda,xpu,hpu,cpu}
                        设备类型。
  --served-model-name SERVED_MODEL_NAME
                        覆盖OpenAI API服务器v1/models端点返回的模型名称。
  --chat-template CHAT_TEMPLATE
                        内置聊天模板名称或聊天模板文件的路径。仅用于兼容OpenAI API的服务器。
  --is-embedding        是否将CausalLM用作嵌入模型。
  --revision REVISION   使用的具体模型版本。可以是分支名、标签名或提交ID。未指定时，使用默认版本。
  --mem-fraction-static MEM_FRACTION_STATIC
                        用于静态分配（模型权重和KV缓存内存池）的内存比例。如果遇到内存不足错误，请使用较小的值。
  --max-running-requests MAX_RUNNING_REQUESTS
                        正在运行的最大请求数量。
  --max-total-tokens MAX_TOTAL_TOKENS
                        内存池中的最大token数量。如果未指定，将根据内存使用比例自动计算。此选项通常用于开发和调试目的。
  --chunked-prefill-size CHUNKED_PREFILL_SIZE
                        分块预填充中每个块的最大token数量。设置为-1表示禁用分块预填充。
  --max-prefill-tokens MAX_PREFILL_TOKENS
                        预填充批次中的最大token数量。实际限制将是此值和模型最大上下文长度之间的较大值。
  --schedule-policy {lpm,random,fcfs,dfs-weight}
                        请求的调度策略。
  --schedule-conservativeness SCHEDULE_CONSERVATIVENESS
                        调度策略的保守程度。较大的值意味着更保守的调度。如果经常看到请求被撤回，请使用较大的值。
  --cpu-offload-gb CPU_OFFLOAD_GB
                        为CPU卸载保留的RAM GB数。
  --prefill-only-one-req PREFILL_ONLY_ONE_REQ
                        如果为true，则每次预填充仅处理一个请求。
  --tensor-parallel-size TENSOR_PARALLEL_SIZE, --tp-size TENSOR_PARALLEL_SIZE
                        张量并行大小。
  --stream-interval STREAM_INTERVAL
                        流式传输的间隔（或缓冲区大小），按token长度计算。较小的值使流式传输更平滑，而较大的值提高吞吐量。
  --stream-output       是否作为一系列不连续的段输出。
  --random-seed RANDOM_SEED
                        随机种子。
  --constrained-json-whitespace-pattern CONSTRAINED_JSON_WHITESPACE_PATTERN
                        JSON约束输出中允许的语法空白的正则表达式模式。例如，要允许模型生成连续的空格，请将模式设置为[\n\t ]*
  --watchdog-timeout WATCHDOG_TIMEOUT
                        设置看门狗超时时间（秒）。如果前向批处理花费的时间超过此值，服务器将崩溃以防止挂起。
  --download-dir DOWNLOAD_DIR
                        模型下载目录。
  --base-gpu-id BASE_GPU_ID
                        开始分配GPU的基础GPU ID。在单台机器上运行多个实例时很有用。
  --log-level LOG_LEVEL
                        所有记录器的日志级别。
  --log-level-http LOG_LEVEL_HTTP
                        HTTP服务器的日志级别。如果没有设置，默认重用--log-level。
  --log-requests        记录所有请求的输入和输出。
  --show-time-cost      显示自定义标记的时间成本。
  --enable-metrics      启用日志Prometheus指标。
  --decode-log-interval DECODE_LOG_INTERVAL
                        解码批次的日志间隔。
  --api-key API_KEY     设置服务器的API密钥。也用于兼容OpenAI API的服务器。
  --file-storage-pth FILE_STORAGE_PTH
                        后端文件存储的路径。
  --enable-cache-report
                        返回openai请求中usage.prompt_tokens_details中的缓存token数量。
  --data-parallel-size DATA_PARALLEL_SIZE, --dp-size DATA_PARALLEL_SIZE
                        数据并行大小。
  --load-balance-method {round_robin,shortest_queue}
                        数据并行性的负载均衡策略。
  --expert-parallel-size EXPERT_PARALLEL_SIZE, --ep-size EXPERT_PARALLEL_SIZE
                        专家并行大小。
  --dist-init-addr DIST_INIT_ADDR, --nccl-init-addr DIST_INIT_ADDR
                        初始化分布式后端的主机地址（例如`192.168.0.2:25000`）。
  --nnodes NNODES       节点数量。
  --node-rank NODE_RANK
                        节点排名。
  --json-model-override-args JSON_MODEL_OVERRIDE_ARGS
                        用于覆盖默认模型配置的JSON字符串格式的字典。
  --lora-paths [LORA_PATHS ...]
                        LoRA适配器列表。可以提供str格式的路径列表，或{名称}={路径}的重命名路径格式。
  --max-loras-per-batch MAX_LORAS_PER_BATCH
                        运行批次中的最大适配器数量，包括仅基础模型的请求。
  --lora-backend LORA_BACKEND
                        选择多LoRA服务的内核后端。
  --attention-backend {flashinfer,triton,torch_native}
                        选择注意力层的内核。
  --sampling-backend {flashinfer,pytorch}
                        选择采样层的内核。
  --grammar-backend {xgrammar,outlines}
                        选择语法引导解码的后端。
  --enable-flashinfer-mla
                        启用FlashInfer MLA优化
  --speculative-algorithm {EAGLE}
                        推测算法。
  --speculative-draft-model-path SPECULATIVE_DRAFT_MODEL_PATH
                        草稿模型权重的路径。可以是本地文件夹或Hugging Face仓库ID。
  --speculative-num-steps SPECULATIVE_NUM_STEPS
                        在推测解码中从草稿模型采样的步骤数。
  --speculative-num-draft-tokens SPECULATIVE_NUM_DRAFT_TOKENS
                        在推测解码中从草稿模型采样的token数。
  --speculative-eagle-topk {1,2,4,8}
                        在eagle2每步中从草稿模型采样的token数。
  --enable-double-sparsity
                        启用双重稀疏性注意力
  --ds-channel-config-path DS_CHANNEL_CONFIG_PATH
                        双重稀疏性通道配置的路径
  --ds-heavy-channel-num DS_HEAVY_CHANNEL_NUM
                        双重稀疏性注意力中的重型通道数量
  --ds-heavy-token-num DS_HEAVY_TOKEN_NUM
                        双重稀疏性注意力中的重型token数量
  --ds-heavy-channel-type DS_HEAVY_CHANNEL_TYPE

```
