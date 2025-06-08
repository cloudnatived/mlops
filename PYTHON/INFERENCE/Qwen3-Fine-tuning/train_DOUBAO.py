import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab
from tqdm import tqdm

os.environ["SWANLAB_PROJECT"] = "qwen3-sft-medical"
PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 2048

swanlab.config.update({
    "model": "Qwen/Qwen3-1.7B",
    "prompt": PROMPT,
    "data_max_length": MAX_LENGTH,
})

def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    添加数据验证和错误处理
    """
    messages = []
    missing_fields = 0
    
    print(f"开始处理数据集: {origin_path}")
    with open(origin_path, "r") as file:
        lines = file.readlines()
        for i, line in enumerate(tqdm(lines, desc="处理数据")):
            try:
                data = json.loads(line)
                # 验证必要字段
                if 'question' not in data or 'think' not in data or 'answer' not in data:
                    missing_fields += 1
                    continue
                    
                input_text = data["question"]
                output = f"<think>{data['think']}</think> \n {data['answer']}"
                message = {
                    "instruction": PROMPT,
                    "input": input_text,
                    "output": output,
                }
                messages.append(message)
            except json.JSONDecodeError as e:
                print(f"Line {i}: JSON解析错误 - {e}")
    
    if missing_fields > 0:
        print(f"警告: {missing_fields}条记录缺少必要字段，已跳过")
    
    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in tqdm(messages, desc="保存数据"):
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
    
    print(f"数据集处理完成，保存至: {new_path}")

def process_func(example):
    """
    将数据集进行预处理
    """ 
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    # 截断处理
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def predict(messages, model, tokenizer, max_length=2048, temperature=0.7, top_p=0.9):
    """
    改进的预测函数，支持更多生成参数配置
    """
    device = next(model.parameters()).device
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    # 生成参数配置
    generation_config = {
        "max_new_tokens": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            **generation_config
        )
    
    # 提取生成的内容（不包括输入）
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# 在modelscope上下载Qwen模型到本地目录下
print("开始下载模型...")
model_dir = snapshot_download("Qwen/Qwen3-1.7B", cache_dir="/root/autodl-tmp/", revision="master")
print(f"模型下载完成，保存在: {model_dir}")

# Transformers加载模型权重
print("开始加载模型...")
tokenizer = AutoTokenizer.from_pretrained(
    "/root/autodl-tmp/Qwen/Qwen3-1.7B", 
    use_fast=False, 
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/Qwen/Qwen3-1.7B", 
    device_map="auto", 
    torch_dtype=torch.bfloat16,        # DOUBAO改进，使用bf16精度
    low_cpu_mem_usage=True,            # 减少加载时的CPU内存使用
    trust_remote_code=True
)

# 启用梯度检查点和优化设置
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
print("模型加载完成")

# 加载、处理数据集和测试集
train_dataset_path = "train.jsonl"
test_dataset_path = "val.jsonl"

train_jsonl_new_path = "train_format.jsonl"
test_jsonl_new_path = "val_format.jsonl"

# 处理数据集
if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

# 加载数据集
print("开始加载和处理训练数据...")
train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names, num_proc=4)
print(f"训练数据加载完成，共{len(train_dataset)}条样本")

print("开始加载和处理验证数据...")
eval_df = pd.read_json(test_jsonl_new_path, lines=True)
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names, num_proc=4)
print(f"验证数据加载完成，共{len(eval_dataset)}条样本")

# 设置训练参数
args = TrainingArguments(
    output_dir="/root/autodl-tmp/output/Qwen3-1.7B",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,        # 增加梯度累积步数以增大有效batch size
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    num_train_epochs=3,                   # 增加训练轮次
    save_steps=400,
    learning_rate=5e-5,                   # 调整学习率
    weight_decay=0.01,                    # 添加权重衰减防止过拟合
    warmup_ratio=0.05,                    # 添加warmup阶段
    fp16=False,                           # 根据硬件支持选择精度
    bf16=True,                            # A100等支持bf16的GPU使用bf16
    save_total_limit=3,                   # 限制保存的检查点数量
    load_best_model_at_end=True,
    report_to="swanlab",
    run_name="qwen3-1.7B-medical",
    dataloader_num_workers=2,
    group_by_length=True,  # 根据序列长度分组，提高训练效率
)

# 创建训练器
print("初始化训练器...")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# 开始训练
print("开始训练模型...")
trainer.train()
print("模型训练完成")

# 保存最终模型
trainer.save_model("/root/autodl-tmp/output/Qwen3-1.7B-final")
print("最终模型已保存")

# 用测试集的前3条，主观看模型
print("开始生成预测结果...")
test_df = pd.read_json(test_jsonl_new_path, lines=True)[:3]
test_text_list = []

for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer)

    response_text = f"""
    Question: {input_value}

    LLM:{response}
    """
    
    test_text_list.append(swanlab.Text(response_text))
    print(response_text)
    print("-" * 50)

# 记录预测结果到SwanLab
swanlab.log({"Prediction": test_text_list})
swanlab.finish()
print("所有任务已完成")
