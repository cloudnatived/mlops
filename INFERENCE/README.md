
# VLLM  SGLang  Ray TensorRT Triton



```

生产环境H200部署DeepSeek 671B 满血版全流程实战（一）：系统初始化
生产环境H200部署DeepSeek 671B 满血版全流程实战（二）：vLLM 安装详解
生产环境H200部署DeepSeek 671B 满血版全流程实战（三）：SGLang 安装详解
生产环境H200部署DeepSeek 671B 满血版全流程实战（四）：vLLM 与 SGLang 的性能大比拼
H200部署DeepSeek R1，SGLang调优性能提升2倍，每秒狂飙4000+ Tokens
猫哥手把手教你基于vllm大模型推理框架部署Qwen3-MoE
基于vLLM v1测试BFloat16 vs FP8 Qwen3-MoE模型吞吐性能的重大发现!
转载: Qwen 3 + KTransformers 0.3 (+AMX) = AI 工作站/PC

```





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


# TensorRT Triton

```
TensorRT&Triton学习笔记(一)：triton和模型部署+client https://zhuanlan.zhihu.com/p/482170985
TensorRT详细入门指北，如果你还不了解TensorRT，过来看看吧  https://blog.csdn.net/IAMoldpan/article/details/117908232
Triton + TensorRT 推理模型部署  https://blog.csdn.net/weixin_39403185/article/details/147105599

CPU版本的启动：
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:22.01-py3 tritonserver --model-repository=/models
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3 tritonserver --model-repository=/models


GPU版本的启动，使用1个gpu：
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:22.01-py3 tritonserver --model-repository=/models
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3 tritonserver --model-repository=/models




```

