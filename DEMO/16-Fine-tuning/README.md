

下载数据集
```
# 下载数据集
meta-math/MetaMathQA
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download meta-math/MetaMathQA --local-dir /Data/MODEL/meta-math/MetaMathQA
hf download meta-math/MetaMathQA --local-dir /Data/MODEL/meta-math/MetaMathQA
hf download meta-math/MetaMathQA --repo-type dataset --local-dir /Data/MODEL/meta-math/MetaMathQA    # 正确的命令

nvidia/OpenMathInstruct-2
export HF_ENDPOINT=https://hf-mirror.com
hf download nvidia/OpenMathInstruct-2 --repo-type dataset --local-dir /Data/MODEL/nvidia/OpenMathInstruct-2    # 正确的命令

AI-MO/NuminaMath-CoT
export HF_ENDPOINT=https://hf-mirror.com
hf download AI-MO/NuminaMath-CoT --repo-type dataset --local-dir /Data/MODEL/AI-MO/NuminaMath-CoT    # 正确的命令

open-r1/OpenR1-Math-220k
export HF_ENDPOINT=https://hf-mirror.com
hf download open-r1/OpenR1-Math-220k --repo-type dataset --local-dir /Data/MODEL/open-r1/OpenR1-Math-220k    # 正确的命令
hf download open-r1/OpenR1-Math-220k --repo-type dataset --local-dir /Data/MODEL/open-r1/OpenR1-Math-220k --local-dir-use-symlinks False --resume-download    # 参数不支持

```



LoRA
```
LoRA

V100上轻松运行。
python3 train_qwen_math_lora.py \
  --model_name_or_path /Data/MODEL/Qwen/Qwen2.5-Math-1.5B-Instruct \
  --data_path /Data/DEMO/Qwen2.5-Math-1.5B-Instruct/data/metamath_40k.jsonl \
  --output_dir /Data/DEMO/Qwen2.5-Math-1.5B-Instruct/output/lora_math \
  --max_seq_len 4096 \
  --per_device_batch_size 1 \
  --grad_accum 8 \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --val_samples 1000 \
  --seed 42



自定义评测题目，写一个文件，比如 eval_prompts.txt：
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
已知 x^2 - 5x + 6 = 0，求 x 的值。
一辆汽车每小时走 60 公里，3 小时后还剩 120 公里，求总路程。
求方程 2x + 7 = 21 的解。
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

python3 train_qwen_math_lora.py \
  --model_name_or_path /Data/MODEL/Qwen/Qwen2.5-Math-1.5B-Instruct \
  --data_path /Data/DEMO/Qwen2.5-Math-1.5B-Instruct/data/metamath_40k.jsonl \
  --output_dir /Data/DEMO/Qwen2.5-Math-1.5B-Instruct/output/lora_math \
  --eval_prompts /Data/DEMO/Qwen2.5-Math-1.5B-Instruct/eval_prompts.txt



python3 LoRA_5_2.py  数学题回答微调（SFT）






++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
```


全量微调

```

# 检查 PyTorch 和 CUDA 环境
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

for i in range(torch.cuda.device_count()):
    print(
        f"GPU {i}:",
        torch.cuda.get_device_name(i),
        "Capability:",
        torch.cuda.get_device_capability(i)
    )
    print("Memory:", torch.cuda.mem_get_info(i))
    print("Supported architectures:", torch.cuda.get_arch_list())
PY

运行 convert_metamath_to_alpaca.py

cd /Data/DEMO/Qwen2.5-Math-1.5B-Instruct
python3 convert_metamath_to_alpaca.py \
  --src /Data/MODEL/meta-math/MetaMathQA/MetaMathQA-395K.json \
  --dst data/metamath_40k.jsonl \
  --num 40000 --seed 42

ds_config_zero2.json
ds_config_zero3.json

bash run_full_finetune.sh
train_full_qwen25_math.py


# 这样也是可以持续运行的。
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \
  train_full_qwen25_math.py \
  --model_name_or_path /Data/MODEL/Qwen/Qwen2.5-Math-1.5B-Instruct \
  --data_path /Data/DEMO/Qwen2.5-Math-1.5B-Instruct/data/metamath_40k.jsonl \
  --output_dir /Data/DEMO/Qwen2.5-Math-1.5B-Instruct/output/full-finetune-test \
  --deepspeed_config /Data/DEMO/Qwen2.5-Math-1.5B-Instruct/ds_config_zero2.json \
  --max_seq_len 64 \
  --per_device_batch_size 1 \
  --grad_accum 16 \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --val_samples 200


# zero3 很快就会OOM。
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \
  train_full_qwen25_math.py \
  --model_name_or_path /Data/MODEL/Qwen/Qwen2.5-Math-1.5B-Instruct \
  --data_path /Data/DEMO/Qwen2.5-Math-1.5B-Instruct/data/metamath_40k.jsonl \
  --output_dir /Data/DEMO/Qwen2.5-Math-1.5B-Instruct/output/full-finetune-test \
  --deepspeed_config /Data/DEMO/Qwen2.5-Math-1.5B-Instruct/ds_config_zero3.json \
  --max_seq_len 64 \
  --per_device_batch_size 1 \
  --grad_accum 16 \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --val_samples 200

对于你的硬件，我建议优先考虑 ZeRO-2 + CPU Optimizer Offload，而不是继续使用当前参数较激进的 ZeRO-3。

原因是：
- 1.5B 模型 FP16 参数总量约 3GB。
- 两张 V100 各有 16GB 显存，模型参数本身并不算特别大。
- ZeRO-2 可以分片优化器状态和梯度，不需要像 ZeRO-3 那样频繁进行参数分片和 AllGather。
- 可以减少 ZeRO-3 参数管理和通信缓冲区带来的复杂性。




```


accelerate
```
# 这个能在V100上跑完
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --num_machines 1 \
  --num_processes 1 \
  --dynamo_backend no \
  --mixed_precision fp16 \
  accelerate_train.py \
  --model_name /Data/MODEL/Qwen/Qwen2.5-Math-1.5B-Instruct \
  --max_steps 4 \
  --max_length 2 \
  --per_device_batch_size 1 \
  --learning_rate 1e-5 \
  --warmup_steps 1 \
  --weight_decay 0.1 \
  --gradient_accumulation_steps 1 \
  --logging_steps 10 \
  --shuffle_buffer 100 \
  --seed 42 \
  --output_dir ./accelerate-qwen-math-full-ft

# 能跑完
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True    # 参数不支持
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --num_machines 1 \
  --num_processes 1 \
  --dynamo_backend no \
  --mixed_precision fp16 \
  accelerate_train.py \
  --model_name /Data/MODEL/Qwen/Qwen2.5-Math-1.5B-Instruct \
  --max_steps 2 \
  --max_length 8 \
  --per_device_batch_size 1 \
  --learning_rate 1e-5 \
  --warmup_steps 1 \
  --weight_decay 0.1 \
  --gradient_accumulation_steps 1 \
  --logging_steps 10 \
  --shuffle_buffer 100 \
  --seed 42 \
  --output_dir ./accelerate-qwen-math-full-ft
```



之前的

```



finetune_test/
├── configs/
│   ├── gemma2_2b_config.yaml
│   ├── gemma2_9b_config.yaml
│   └── gemma2_27b_config.yaml
├── data/
│   └── sample_data.jsonl  # 你的训练数据
├── scripts/
│   ├── train_full.py      # 全量微调
│   ├── train_lora.py      # LoRA微调
│   ├── train_qlora.py     # QLoRA微调
│   ├── train_ptuning.py   # P-Tuning微调
│   └── utils.py           # 工具函数
└── run_experiments.sh     # 批量运行脚本

# 在 train_lora.py 中添加
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# 使用示例
config = load_config("configs/gemma2_2b_config.yaml")
lora_config = config['lora']['lora_config']
training_args = TrainingArguments(**config['lora']['training'])

google/gemma-4-E2B
huggingface-cli download google/gemma-4-E2B --local-dir /Data/MODEL/google/gemma-4-E2B

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download google/gemma-2-2b --local-dir /Data/MODEL/google/gemma-2-2b

/Data/MODEL/google/gemma-4-E2B/    10.3 GB


https://unsloth.ai/docs/zh/mo-xing/gemma-4/train
docker run -it -d --shm-size=8G --gpus all --network host --cap-add SYS_ADMIN -v /Data:/Data nvcr.io/nvidia/pytorch:23.10-py3     # 推荐使用这个镜像
pip download -r vit_0.py.requirements.txt --dest /Data/whl -v   # 把requirements.txt文件中列出的包全都先下载到/Data/IMAGES/whl目录，不安装。
pip install -r vit_0.py.requirements.txt --find-links=/Data/whl --no-index #从/Data/IMAGES/whl寻找安装包安装，不要連接到 PyPI。


pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

pip download transformers datasets peft --dest /Data/whl -v
pip install transformers datasets peft --find-links=/Data/whl --no-index
python3 train_full.py


docker run -it -d --shm-size=8G --gpus all --network host --cap-add SYS_ADMIN -v /Data:/Data nvcr.io/nvidia/pytorch:25.09-py3
/Data/DEMO/14-LLaMA-Factory/finetune_test/scripts
python3 train_full.py



```
