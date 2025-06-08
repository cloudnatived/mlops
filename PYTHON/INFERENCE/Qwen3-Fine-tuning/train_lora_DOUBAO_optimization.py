import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
import os

PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 2048

def dataset_jsonl_transfer(origin_path, new_path):
    """将原始数据集转换为大模型微调所需数据格式的新数据集"""
    messages = []
    
    print(f"开始处理数据集: {origin_path}")
    with open(origin_path, "r") as file:
        for line in file:
            try:
                data = json.loads(line)
                input_text = data["question"]
                think = data["think"]
                answer = data["answer"]
                output = f"<|FunctionCallBegin|>{think}superscript: \n {answer}"
                
                message = {
                    "instruction": PROMPT,
                    "input": input_text,
                    "output": output,
                }
                messages.append(message)
            except Exception as e:
                print(f"处理行时出错: {e}")
    
    print(f"保存处理后的数据集到: {new_path}")
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

def process_func(example):
    """将数据集进行预处理"""
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def predict(messages, model, tokenizer, max_length=2048, temperature=0.7, top_p=0.9):
    """增强的预测函数，支持更多生成参数配置"""
    device = next(model.parameters()).device
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
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
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# 下载模型
print("开始下载模型...")
model_dir = snapshot_download("Qwen/Qwen3-1.7B", cache_dir="./", revision="master")
print(f"模型下载完成，保存在: {model_dir}")

# 加载模型和分词器
print("开始加载模型和分词器...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "./Qwen/Qwen3-1.7B", 
        use_fast=False, 
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "./Qwen/Qwen3-1.7B", 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.enable_input_require_grads()
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    raise

# 配置LoRA（移除enable_lora参数）
print("配置LoRA参数高效微调...")
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    fan_in_fan_out=False,
    # 移除enable_lora参数
)

model = get_peft_model(model, config)
print("LoRA配置完成")
model.print_trainable_parameters()

# 加载和处理数据集
print("开始加载和处理数据集...")
train_dataset_path = "train.jsonl"
test_dataset_path = "val.jsonl"
train_jsonl_new_path = "train_format.jsonl"
test_jsonl_new_path = "val_format.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

# 处理训练数据
train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names, num_proc=4)
print(f"训练数据加载完成，共{len(train_dataset)}条样本")

# 处理验证数据
eval_df = pd.read_json(test_jsonl_new_path, lines=True)
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names, num_proc=4)
print(f"验证数据加载完成，共{len(eval_dataset)}条样本")

# 设置训练参数
print("配置训练参数...")
args = TrainingArguments(
    output_dir="./output/Qwen3-1.7B",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=400,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=False,
    bf16=True,
    save_total_limit=3,
    load_best_model_at_end=True,
    run_name="qwen3-1.7B-medical",
    dataloader_num_workers=2,
    group_by_length=True,
    gradient_checkpointing=True,
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
try:
    trainer.train()
    print("模型训练完成")
except KeyboardInterrupt:
    print("训练被用户中断")
except Exception as e:
    print(f"训练过程出错: {e}")
    trainer.save_model("./output/error_checkpoint")
    raise

# 保存模型
print("保存训练好的模型...")
trainer.save_model("./output/Qwen3-1.7B-final")
model.save_pretrained("./output/Qwen3-1.7B-lora")
print("模型保存完成")

# 模型预测示例
print("开始生成预测结果...")
test_df = pd.read_json(test_jsonl_new_path, lines=True)[:3]

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
    
    print(response_text)
    print("-" * 50)

print("所有任务已完成")
