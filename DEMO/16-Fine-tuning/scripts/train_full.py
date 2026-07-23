import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from utils import load_dataset, load_tokenizer, load_model_for_full
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b")
    parser.add_argument("--data_path", type=str, default="./data/sample_data.jsonl")
    parser.add_argument("--output_dir", type=str, default="./outputs/full")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()
    
    # 加载tokenizer和模型
    tokenizer = load_tokenizer(args.model_name)
    model = load_model_for_full(args.model_name)
    
    # 加载数据集
    dataset = load_dataset(args.data_path, tokenizer, args.max_length)
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言模型
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        fp16=True,  # V100使用fp16
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="no",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",  # 不自动上报到wandb
        gradient_checkpointing=True,
        optim="adamw_torch",
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
    
    # 保存最终模型
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✅ 全量微调完成！模型保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
