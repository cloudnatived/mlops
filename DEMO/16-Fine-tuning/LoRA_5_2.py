# 数学题回答微调（SFT）
import torch
import os
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
# AutoTokenizer：把文本转成模型能处理的 token ids
# AutoModelForCausalLM：加载自回归语言模型（因果LM）
# TrainingArguments：训练配置参数
from trl import SFTTrainer    #  Hugging Face 的监督微调训练器，适合纯文本聊天模型的指令微调
from peft import LoraConfig    # 参数高效微调（LoRA）配置，减少训练参数量，节省显存
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"    # 设置允许 PyTorch 的缓存分配器按需扩展内存段，而不是分配固定大小的块，从而减少碎片。
#from accelerate import Accelerator

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # 单卡，避免 DataParallel 复制
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
model_name = "/Data/MODEL/Qwen/Qwen2.5-Math-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #torch_dtype=torch.bfloat16,
    torch_dtype=torch.float16, # 1. V100 强烈建议用 float16，不要用 bfloat16（Volta 无原生 BF16 硬件）
    attn_implementation="eager"
)
model.config.use_cache = False

# ========== 加载四个数据集并统一为 question / answer 格式 ==========

metamath = load_dataset("/Data/MODEL/meta-math/MetaMathQA", split="train")
metamath = metamath.map(lambda x: {
    "question": x["query"],
    "answer": x["response"]
}, remove_columns=metamath.column_names)

openmath = load_dataset("/Data/MODEL/nvidia/OpenMathInstruct-2", split="train_1M")
openmath = openmath.map(lambda x: {
    "question": x["problem"],
    "answer": x["generated_solution"]
}, remove_columns=openmath.column_names)

numina = load_dataset("/Data/MODEL/AI-MO/NuminaMath-CoT", split="train")
numina = numina.map(lambda x: {
    "question": x["problem"],
    "answer": x["solution"]
}, remove_columns=numina.column_names)

r1_data = load_dataset("/Data/MODEL/open-r1/OpenR1-Math-220k", split="train")
r1_data = r1_data.map(lambda x: {
    "question": x["problem"],
    "answer": x["solution"]
}, remove_columns=r1_data.column_names)

combined = concatenate_datasets([metamath, openmath, numina, r1_data])
combined = combined.shuffle(seed=42)
print(f"合并后数据集大小: {len(combined)}")

# 抽样
#MAX_SAMPLES = 50000
#MAX_SAMPLES = 10000                         # 先小规模验证
MAX_SAMPLES = 10                         # 先小规模验证
if len(combined) > MAX_SAMPLES:
    combined = combined.select(range(MAX_SAMPLES))
    print(f"已抽样至: {len(combined)} 条")

# ========== 预先映射出 text 列（替代 formatting_func） ==========
# 使用 num_proc=1，避免多进程下 tokenizer/template 状态异常
def add_text(example):
    # 防御：确保 question 和 answer 是字符串
    q = str(example["question"]) if example["question"] is not None else ""
    a = str(example["answer"]) if example["answer"] is not None else ""
    messages = [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

combined = combined.map(add_text, num_proc=1)
# 映射完成后，只保留 text 列，避免 SFTTrainer 自动转换格式
combined = combined.select_columns(["text"])
print("示例 text:", combined[0]["text"][:200])

# ========== LoRA 配置与训练 ==========

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir="./qwen-math-lora",
    #per_device_train_batch_size=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,         # 等效 batch = 32
    num_train_epochs=2,
    learning_rate=1e-4,                    # LoRA 常用 1e-4 ~ 2e-4
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    bf16=False,                         # 配合 float16
    fp16=True,                          # V100 用 fp16       
    ddp_find_unused_parameters=False,        # 多卡时建议加
    gradient_checkpointing=True,           # 2. 打开梯度检查点（用时间换显存，最有效）
    #gradient_checkpointing=False,           # 系统内存是瓶颈，关掉它
    gradient_checkpointing_kwargs={"use_reentrant": False},  # ← 推荐加这行
    report_to="none",
)

# SFTTrainer 会自动应用 chat template、只在 assistant 部分计算 loss
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=combined,
    peft_config=peft_config,
    dataset_text_field="text",          # 关键：直接指定 text 列
    #max_seq_length=4096,                # max_length：建议 2048~4096。很多数学数据因为截断导致答案被切掉，等于喂了错误监督信号。
    max_seq_length=2048,
    packing=False,                      # 先保持 False，稳定优先
    dataset_num_proc=1,                 # 已预处理，无需多进程
)

print(f"Model device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'single GPU'}")
#print(torch.cuda.memory_summary())

trainer.train()
trainer.save_model("./qwen-math-lora")
