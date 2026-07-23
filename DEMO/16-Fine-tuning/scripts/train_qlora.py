import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from utils import (
    load_dataset, 
    load_tokenizer, 
    load_model_for_qlora,
    get_gemma_target_modules
)
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-27b")
    parser.add_argument("--data_path", type=str, default="./data/sample_data.jsonl")
    parser.add_argument("--output_dir", type=str, default="./outputs/qlora")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (QLoRA建议用更大的r)")
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_all", action="store_true", default=True)
    parser.add_argument("--use_double_quant", action="store_true", default=True)
    parser.add_argument("--use_cpu_offload", action="store_true", help="是否使用CPU Offload（27B模型需要）")
    args = parser.parse_args()
    
    # 加载tokenizer
    tokenizer = load_tokenizer(args.model_name)
    
    # 配置QLoRA
    if args.target_all:
        target_modules = get_gemma_target_modules(args.model_name)
    else:
        target_modules = ["q_proj", "v_proj"]
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    
    print(f"🔧 QLoRA配置:")
    print(f"  - Rank: {args.lora_r}")
    print(f"  - Alpha: {args.lora_alpha}")
    print(f"  - Target modules: {target_modules}")
    print(f"  - Double Quant: {args.use_double_quant}")
    
    # 加载模型（使用QLoRA）
    model = load_model_for_qlora(args.model_name, lora_config)
    
    # 如果启用CPU Offload（针对27B模型）
    if args.use_cpu_offload:
        print("⚠️  启用CPU Offload（速度会变慢，但能跑通大模型）")
        # 这里可以在TrainingArguments中设置
        # 或在模型加载时设置 device_map="auto" 配合 offload_folder
    
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
        gradient_checkpointing=True,  # QLoRA推荐开启
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        dataloader_pin_memory=False,
        # CPU Offload相关（如果启用）
        **({"optim": "adamw_torch", "optim_args": "offload_optimizer=True"} if args.use_cpu_offload else {})
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
    
    # 保存QLoRA适配器
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✅ QLoRA微调完成！适配器保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
