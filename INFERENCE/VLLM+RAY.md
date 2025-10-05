


```


9. 高性能推理引擎vLLM单机部署
vLLM常见问题
1. CUDA Out of Memory: 模型权重或KV Cache超出显存。解决方法是减小batch size、序列长度，或使用更强的量化。
2. 模型架构不识别: Transformers库版本过低，需要升级。
3. P2P能力警告: V100 PCIE版本可能不支持P2P，添加--disable-custom-all-reduce参数可规避此问题。

https://zhuanlan.zhihu.com/p/1916898243423500022    vLLM参数详细说明
https://blog.csdn.net/baiyipiao/article/details/141930442    vllm常用参数总结
https://www.studywithgpt.com/zh-cn/tutorial/zl0s7e    使用Docker部署vLLM

确保PyTorch版本≤2.1.2，CUDA版本为11.8，V100-PCIE-16GB
transformers>=4.40.0  # Qwen3需要≥4.40

对于 ​​Tesla V100-PCIE-16GB​​，以下是经过验证的兼容CUDA容器版本：
​​1. 推荐版本矩阵​​
容器版本	                               CUDA版本	PyTorch	   TensorRT	  兼容性	 推荐度
nvcr.io/nvidia/pytorch:23.10-py3	      11.8	   2.1.2	    8.6.1	    ✅ 最佳	⭐⭐⭐⭐⭐
nvcr.io/nvidia/tensorrt:23.09-py3	      11.8	   -	        8.6.      ✅ 优秀	⭐⭐⭐⭐
nvidia/cuda:11.8.0-runtime-ubuntu20.04	11.8	   需安装	     -	       ✅ 稳定	 ⭐⭐⭐
nvcr.io/nvidia/pytorch:22.12-py3	      11.7	   1.14	      8.5.3   	✅ 良好	⭐⭐⭐

不兼容版本警告​​
​​避免以下版本​​：
容器版本	                         问题
nvcr.io/nvidia/pytorch:24.0*	    CUDA 12.4，V100支持不完善
nvidia/cuda:12.*	                需要驱动535+，可能不兼容旧系统
任何包含PyTorch                      2.2+的版本	已弃用计算能力7.0

# 硬件是 Tesla V100（算力 7.0），但日志中 PyTorch 明确提示：The minimum cuda capability supported by this library is 7.5（最低支持算力 7.5，如 Tesla T4）。
# 当前 vLLM 镜像（v0.10.1.1）内置的 PyTorch 版本过高（推测 ≥2.1），已移除对算力 7.0 的支持。
# v100建议使用vllm/vllm-openai:v0.3.0镜像
# vllm/vllm-openai:v0.3.0	核心：旧版镜像内置 PyTorch 2.0，支持算力 7.0 的 V100 GPU
# --disable-custom-all-reduce	禁用 vLLM 自定义的分布式通信优化（部分新版优化不兼容 V100），确保 NCCL 通信正常

# 适用vllm/vllm-openai:v0.3.0镜像，使用Qwen2.5-1.5B的模型，进行调试

适合V100 16GB的模型
# 推荐模型大小
MODELS=(
    "Qwen2-7B-Instruct"     # 7B参数，适合2个V100
    "Llama-2-7b-chat-hf"    # 7B参数
    "Mistral-7B-Instruct"   # 7B参数
    "Baichuan2-7B-Chat"     # 7B参数
)

# 启动参数优化
OPTIMAL_CONFIG="--tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --dtype float16 \
    --enforce-eager \
    --max-num-batched-tokens 2048"


hub.rat.dev/vllm/vllm-openai:v0.3.0                     # 适用V100-PCIE-16GB型号
hub.rat.dev/lmsysorg/sglang:v0.3.6.post3-cu124          # 适用V100-PCIE-16GB型号，但是sglang的这个版本已移除对算力 7.0 的支持。

hub.rat.dev/vllm/vllm-openai:v0.10.1.1                  # 已移除对算力 7.0 的支持。
m.daocloud.io/vllm/vllm-openai:v0.10.1.1                # 已移除对算力 7.0 的支持。
nvcr.io/nvidia/pytorch:24.05-py3
nvcr.io/nvidia/pytorch:24.02-py3
nvcr.io/nvidia/pytorch:23.10-py3
nvcr.io/nvidia/tensorrt-llm/release:0.21.0rc0
lmsysorg/sglang:v0.4.6.post4-cu124
nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3     # 这些镜像启动，需要带模型目录才能完成启动
nvcr.io/nvidia/tritonserver:25.08-pyt-python-py3
nvcr.io/nvidia/tritonserver:25.08-trtllm-python-py3
m.daocoud.io/lmsysorg/sglang:v0.5.0rc2-cu129-gb200

# 如果使用非vllm-openai容器镜像部署的话，需要在容器镜像内安装vllm
pip install vllm==0.3.0
pip install "numpy<2" "transformers<4.40"
pip install "numpy<2" "transformers<4.40" "torch==2.1.2"  # 在主节点，从节点上都执行
对应的torch版本：torch-2.1.2-cp310-cp310-manylinux1_x86_64.whl
对应的vllm版本：vllm-0.3.0-cp310-cp310-manylinux1_x86_64.whl
部署vllm的时候会需要triton这个依赖，版本是：triton-2.1.0-0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl


Qwen/Qwen3-32B 
在V100-PCIE-16GB上使用ollama可以部署Qwen/Qwen3-32B进行推理，但是无法使用新版的vllm部署Qwen/Qwen3-32B。
1. ​​FP16精度部署​
模型规模	参数量	所需显存	你的配置支持情况
Qwen2-32B​​	32B	~64GB	✅ ​​最佳选择​​（4卡并行）

Qwen2-32B（FP16） - 最佳平衡​​
# 使用4张GPU进行张量并行
python -m torch.distributed.run --nproc_per_node=4 --nnodes=3 your_deploy_script.py --model-name Qwen/Qwen2-32B  #使用torch分布式部署Qwen2-32B的命令参考。

​​所需显存​​：~64GB
​​使用GPU​​：4张V100
​​剩余资源​​：2张GPU备用或部署其他服务

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-14B --local-dir /Data/Qwen/Qwen2.5-14B    # 下载模型

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-7B --local-dir /Data/Qwen/Qwen2.5-7B      # 下载模型

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir /Data/Qwen/Qwen2.5-1.5B    # 使用较小模型，进行多机多卡的分布式部署

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-14B --local-dir /Data/Qwen/Qwen3-14B


# 参数小的模型能执行完成，大约5分钟完成，使用非gated替代模型<200b>，Falcon-7BB (Apache 2.0许可)，
# 代替meta-llama/Llama-2-7b，因为meta-llama/Llama-2-7b在huggingface需要登录，而且登录之后，仍无法下载，通过huggingface镜像网站也无法下载
# 创建输出目录
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download meta-llama/Llama-2-7b --local-dir /Data/meta-llama/Llama-2-7b      # 下载huggingface上的meta-llama模型文件，需要登录。
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir /Data/mistralai/mmistral-7b   # 使用非gated替代模型<200b>，Mistral-7B (性能优于Llama2-7B)
huggingface-cli download tiiuae/falcon-7b --local-dir /Data/tiiuae/falcon-7b                 # 使用非gated替代模型<200b>，Falcon-7B (Apache 2.0许可)

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download daryl149/llama-2-7b-chat-hf --local-dir /Data/meta-llama/Llama-2-7b
huggingface-cli download daryl149/llama-2-7b-chat-hf --local-dir /Data/llama-2-7b

https://www.modelscope.cn/models/LLM-Research/llama-2-7b  #  下载，llama-2-7b
git clone LLM-Research/llama-2-7b
git clone https://www.modelscope.cn/LLM-Research/llama-2-7b.git


VLLM
# 尝试，参数参考
# 使用nvcr.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04，在一个cpu节点上运行vllm服务。--network host咋个模式下，不需要要设置-p参数，参数参考
docker run -it -d --network host --cap-add SYS_ADMIN -v /Data/Qwen/Qwen3-32B:/Data/Qwen/Qwen3-32B nvcr.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04
a2e72019f4d4
pip install vllm

docker pull hub.rat.dev/vllm/vllm-openai:v0.10.1.1    # V100-PCIE-16GB上，算力不匹配，torch版本不匹配，在本次部署中无用
docker pull m.daocloud.io/vllm/vllm-openai:v0.10.1.1  # V100-PCIE-16GB上，算力不匹配，torch版本不匹配，在本次部署中无用
docker pull m.daocloud.io/vllm/vllm-openai:v0.3.0     # 算力 7.0 的支持，V100-PCIE-16GB上算力和torch版本匹配
docker pull hub.rat.dev/vllm/vllm-openai:v0.3.0       # 算力 7.0 的支持，V100-PCIE-16GB上算力和torch版本匹配

# 运行vllm服务的参数参考
["python3" "-m" "vllm.entrypoints.openai.api_server"]
python3 -m vllm.entrypoints.openai.api_server --model /Data/Qwen/Qwen3-32B \
  --served-model-name Qwen3_32B --host 0.0.0.0 --port 6800 --block-size 16 \
  --pipeline-parallel-size 2 --trust-remote-code --enforce-eager \
  --distributed-executor-backend ray --ray-cluster-address 172.18.6.64:6379

# hhub.rat.dev/vllm/vllm-openai:v0.3.0，在单个节点上，使用2个V100-PCIE-16GB，部署
# 这个是在GPU节点上可以启动的，在172.18.8.210上启动了，云主机配置是8C，32G，2个V100-PCIE-16GB

# 单机双卡V100-PCIE-16GB部署，可以成功启动 Qwen2.5-1.5B，成功启动：
docker run --rm \
  --gpus all \
  --net=host --pid=host --ipc=host --privileged --shm-size=4G \
  -v /Data/Qwen/Qwen2.5-1.5B:/model \
  -e VLLM_HOST_IP=172.18.8.210 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_SWAP_SPACE=1 \
  hub.rat.dev/vllm/vllm-openai:v0.3.0 \
  --model /model \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --dtype half \
  --swap-space 1 \
  --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 131072 \
  --trust-remote-code


# 单机双卡V100-PCIE-16GB部署，加载Qwen2.5-14B时，成功启动：
docker run --rm \
  --gpus all \
  --net=host --pid=host --ipc=host --privileged --shm-size=4G \
  -v /Data/Qwen/Qwen2.5-14B:/model \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e RAY_memory_usage_threshold=0.99 \
  -e RAY_memory_monitor_refresh_ms=0 \
  hub.rat.dev/vllm/vllm-openai:v0.3.0 \
  --model /model \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --dtype half \
  --swap-space 4 \
  --gpu-memory-utilization 0.995 \
  --max-model-len 400 \
  --max-num-batched-tokens 400 \
  --max-num-seqs 16 \
  --max-paddings 16 \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --enforce-eager


# 单机双卡V100-PCIE-16GB部署，加载Qwen2.5-7B时，成功启动：
docker run --rm \
  --net=host --pid=host --ipc=host --privileged --shm-size=4G \
  -v /Data/Qwen/Qwen2.5-7B:/model \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e RAY_memory_usage_threshold=0.99 \
  -e RAY_memory_monitor_refresh_ms=0 \
  hub.rat.dev/vllm/vllm-openai:v0.3.0 \
  --model /model \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --dtype half \
  --swap-space 4 \
  --gpu-memory-utilization 0.995 \
  --max-model-len 400 \
  --max-num-batched-tokens 400 \
  --max-num-seqs 16 \
  --max-paddings 16 \
  --trust-remote-code \
  --disable-custom-all-reduce


# 单机双卡V100-PCIE-16GB部署，加载llama-2-7b时，成功启动：
docker run --rm \
  --gpus all \
  --net=host --pid=host --ipc=host --privileged --shm-size=4G \
  -v /Data/llama-2-7b:/model \
  -e VLLM_HOST_IP=172.18.8.208 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_SWAP_SPACE=1 \
  hub.rat.dev/vllm/vllm-openai:v0.3.0 \
  --model /model \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --dtype half \
  --swap-space 1 \
  --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 4096



1. 服务健康状态检查
在发送具体推理请求前，可以先确认服务是否已成功启动并准备就绪。
vLLM 健康检查
curl -v http://172.18.8.210:8000/health

Triton Server 健康检查
curl -v http://172.18.8.208:8000/v2/health/ready

Ollama 模型列表检查
curl http://127.0.0.1:11434/api/tags


# 检查服务是否启动
curl -v http://172.18.8.208:8000/health

# 测试API请求
curl http://172.18.8.208:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 5000}'  # 测试长序列

# 模型列表 - 验证模型加载，查询模型信息
curl http://172.18.8.208:8000/v1/models

# 对话（Chat Completion）​，这个需要设置，chat_template.jinja，否则无法访问。
curl http://172.18.8.208:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/model",
        "messages": [
        {"role": "user", "content": "你好，介绍一下你自己"}
        ],
        "max_tokens": 200
    }'

# 生vLLM 文本生成 (Completions API)
curl http://172.18.8.208:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/model",
        "prompt": "如何学习深度学习？",
        "max_tokens": 150,
        "temperature": 0.8
    }' | jq .  # 用jq美化输出

# 使用 vLLM API访问：
curl http://172.18.8.208:8000/v1/completions -H "Content-Type: application/json" -d '{
  "model": "/model",
  "prompt": "Hello world",
  "max_tokens": 128
}'


# 问题集，及解决线索：
# 会出现很多报错。需要不断调整参数。
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.25 GiB. GPU 0 has a total capacty of 15.77 GiB of which 281.12 MiB is free. 
Including non-PyTorch memory, this process has 15.49 GiB memory in use. 
Of the allocated memory 15.04 GiB is allocated by PyTorch, and 4.72 MiB is reserved by PyTorch but unallocated. 
If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  
See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
(RayWorkerVllm pid=1756) WARNING 09-11 02:01:17 custom_all_reduce.py:44] Custom allreduce is disabled because your platform lacks GPU P2P capability. 
To slience this warning, specifydisable_custom_all_reduce=True explicitly.

# GPU blocks: 0
ValueError: No available memory for the cache blocks.
→ 权重本身已占满 2×V100-16GB，KV-cache 一块都分不到（0 blocks）。

Qwen2.5-14B 半精度权重 ≈ 28 GB，
2×16 GB = 32 GB → 只剩 < 4 GB 用于 KV-cache，
即使 gpu_memory_utilization=0.98 也 无法容纳最小 cache。

# V100-PCIE-16GB上，无法部署Qwen2.5-32B模型。云主机配置8C，32G，2个V100-PCIE-16GB。
# 错误信息。
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.31 GiB. GPU 0 has a total capacty of 15.77 GiB of which 1.76 GiB is free. 
Including non-PyTorch memory, this process has 14.00 GiB memory in use. 
Of the allocated memory 13.51 GiB is allocated by PyTorch, and 29.85 MiB is reserved by PyTorch but unallocated. 
If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
(RayWorkerVllm pid=100969) WARNING 09-10 05:13:58 custom_all_reduce.py:44] Custom allreduce is disabled because your platform lacks GPU P2P capability. To slience this warning, 
specifydisable_custom_all_reduce=True explicitly.

# 错误信息。
ValueError: The checkpoint you are trying to load has model type `qwen3` but Transformers does not recognize this architecture. 
This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.

# 错误信息。
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 36.00 MiB. GPU 0 has a total capacty of 15.77 GiB of which 22.25 MiB is free. 
Including non-PyTorch memory, this process has 15.74 GiB memory in use. 
Of the allocated memory 15.29 GiB is allocated by PyTorch, and 6.66 MiB is reserved by PyTorch but unallocated. 
If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  
See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
(RayWorkerVllm pid=100129) WARNING 09-10 05:09:56 custom_all_reduce.py:44] Custom allreduce is disabled because your platform lacks GPU P2P capability. 
To slience this warning, specifydisable_custom_all_reduce=True explicitly.









10. 使用vllm+ray集群，进行多机多卡的部署测试
使用vllm+ray集群进行多机多卡的部署测试。3台GPU服务器。每个GPU服务器配置为2个V100-PCIE-16GB，服务器之间的网络为10G。
# Qwen2.5-1.5B使用4卡，不能使用6卡，因为不能整除。大约5分钟完成加载。
# Qwen2.5-7B使用4卡，未测试。
# Qwen2.5-14B使用4卡，无法完成部署，大概加载模型30分钟后。会出错：ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization` when initializing the engine.
# V100-PCIE-16GB上，无法部署Qwen2.5-14B模型。

# 在命令行里启动vllm的命令，参考：
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
RAY_ADDRESS=172.18.8.210:6379 \
python3 -m vllm.entrypoints.openai.api_server \
  --model /model \
  --tensor-parallel-size 4 \
  --worker-use-ray \
  --host 0.0.0.0 \
  --dtype half \
  --swap-space 1 \
  --enforce-eager \
  --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 131072 \
  --trust-remote-code \
  --disable-custom-all-reduce


# 在172.18.8.208这个节点上启动，并设置172.18.8.208容器里的ray，为head节点：
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
RAY_ADDRESS=172.18.8.208:6379 \
docker run --rm \
  --gpus all \
  --net=host --pid=host --ipc=host --privileged --shm-size=4G \
  -v /Data/llama-2-7b:/model \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_SWAP_SPACE=1 \
  -e RAY_NODE_TYPE=head \
  -e RAY_HEAD_SERVICE_HOST=172.18.8.208 \
  -e RAY_HEAD_SERVICE_PORT=6379 \
  -e RAY_DASHBOARD_PORT=8265 \
  -e VLLM_TENSOR_PARALLEL_SIZE=2 \
  hub.rat.dev/vllm/vllm-openai:v0.6.3 \
  --model /model \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --dtype half \
  --swap-space 1 \
  --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 4096 \
  --disable-custom-all-reduce \
  --worker-use-ray \
  --enforce-eager


# 在172.18.8.209和172.18.8.210节点上启动，并设置172.18.8.209和172.18.8.210容器里的ray，为work节点：
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
RAY_ADDRESS=172.18.8.208:6379 \
docker run --rm \
  --gpus all \
  --net=host --pid=host --ipc=host --privileged --shm-size=4G \
  -v /Data/llama-2-7b:/model \
  -e VLLM_HOST_IP=172.18.8.209 \
  -e RAY_HEAD_SERVICE_PORT=6379 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_SWAP_SPACE=1 \
  -e RAY_NODE_TYPE=worker \
  -e RAY_HEAD_SERVICE_HOST=172.18.8.208 \
  -e RAY_HEAD_SERVICE_PORT=6379 \
  -e RAY_DASHBOARD_PORT=8265 \
  -e VLLM_TENSOR_PARALLEL_SIZE=2 \
  -e RAY_ADDRESS=172.18.8.208:6379 \
  hub.rat.dev/vllm/vllm-openai:v0.6.3 \
  --model /model \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --dtype half \
  --swap-space 1 \
  --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 4096 \
  --disable-custom-all-reduce \
  --worker-use-ray \
  --enforce-eager


# 命令行总结
  --max-num-batched-tokens 131072 \ # 不需要添加
  --gpu-memory-utilization 0.8 \ 
  -e RAY_ADDRESS="172.18.8.208:10001" \
  -e RAY_REDIS_PASSWORD="your_password" \  # RAY密码

vocab_size = 151936 无法被 6 整除，而你现在要求 --tensor-parallel-size 6。
vLLM 的 VocabParallelEmbedding 必须整除，否则会直接断言失败。

# 不支持的参数
--distributed-executor-backend ray       # 参数在vllm-openai:v0.3.0中不支持。
--enable-chunked-prefill False           # 参数在vllm-openai:v0.3.0中不支持。
--ray-cluster-address 172.18.6.64:6379   # 参数在vllm-openai:v0.3.0中不支持




```
