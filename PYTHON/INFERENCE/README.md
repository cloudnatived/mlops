
# VLLM SGLang Ray 



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
  
参考资料：
SGLang部署deepseek-ai/DeepSeek-R1-Distill-Qwen-32B实测比VLLM快30%，LMDeploy比VLLM快50% https://blog.csdn.net/weixin_46398647/article/details/145588854

```
SGLang启动
CUDA_VISIBLE_DEVICES=5,6,7,8 python3 -m sglang.launch_server --model ~/.cache/modelscope/hub/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/ --tp 4 --host 0.0.0.0 --port 8000

VLLM启动
CUDA_VISIBLE_DEVICES=5,6,7,8 vllm serve ~/.cache/modelscope/hub/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/ --tensor-parallel-size 4 --max-model-len 32768 --enforce-eager --served-model-name DeepSeek-R1-Distill-Qwen-32B --host 0.0.0.0

LMDeply启动
CUDA_VISIBLE_DEVICES=5,6,7,8 lmdeploy serve api_server  ~/.cache/modelscope/hub/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/ --tp 4 --server-port 8000 --model-name DeepSeek-R1-Distill-Qwen-32B

```




sglang的官网   https://docs.sglang.ai/backend/server_arguments.html   
  
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
