import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from utils import (
    load_dataset, 
    load_tokenizer, 
    load_model_for_lora,
    get_gemma_target_modules
)
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b")
    parser.add_argument("--data_path", type=str, default="./data/sample_data.jsonl")
    parser.add_argument("--output_dir", type=str, default="./outputs/lora")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_all", action="store_true", help="是否在所有线性层应用LoRA")
    args = parser.parse_args()
    
    # 加载tokenizer
    tokenizer = load_tokenizer(args.model_name)
    
    # 配置LoRA
    if args.target_all:
        target_modules = get_gemma_target_modules(args.model_name)
    else:
        # 仅在Q和V上应用LoRA
        target_modules = ["q_proj", "v_proj"]
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    
    print(f"🔧 LoRA配置:")
    print(f"  - Rank: {args.lora_r}")
    print(f"  - Alpha: {args.lora_alpha}")
    print(f"  - Target modules: {target_modules}")
    print(f"  - 可训练参数: {lora_config.r * len(target_modules) * 2:,}")
    
    # 加载模型
    model = load_model_for_lora(args.model_name, lora_config)
    
    # 加载数据集
    dataset = load_dataset(args.data_path, tokenizer, args.max_length)
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=False,  # LoRA通常不需要
        optim="paged_adamw_8bit",  # 使用8-bit优化器节省显存
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        dataloader_pin_memory=False,
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存LoRA适配器
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✅ LoRA微调完成！适配器保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
