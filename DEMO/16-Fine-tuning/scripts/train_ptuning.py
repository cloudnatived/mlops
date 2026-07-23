import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import PromptEncoderConfig, get_peft_model, TaskType
from utils import load_dataset, load_tokenizer, load_model_for_ptuning
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b")
    parser.add_argument("--data_path", type=str, default="./data/sample_data.jsonl")
    parser.add_argument("--output_dir", type=str, default="./outputs/ptuning")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_virtual_tokens", type=int, default=20, help="P-Tuning虚拟token数量")
    parser.add_argument("--encoder_hidden_size", type=int, default=128, help="编码器隐藏层大小")
    args = parser.parse_args()
    
    # 加载tokenizer
    tokenizer = load_tokenizer(args.model_name)
    
    # 配置P-Tuning
    ptuning_config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.num_virtual_tokens,
        encoder_hidden_size=args.encoder_hidden_size,
        encoder_reparameterization_type="mlp",  # MLP或LSTM
        encoder_dropout=0.1,
        inference_mode=False,
    )
    
    print(f"🔧 P-Tuning配置:")
    print(f"  - 虚拟token数: {args.num_virtual_tokens}")
    print(f"  - 编码器隐藏层: {args.encoder_hidden_size}")
    print(f"  - 可训练参数: {args.num_virtual_tokens * args.encoder_hidden_size:,}")
    
    # 加载模型
    model = load_model_for_ptuning(args.model_name, ptuning_config)
    
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
        gradient_checkpointing=False,
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
    
    # 保存P-Tuning适配器
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✅ P-Tuning微调完成！适配器保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
