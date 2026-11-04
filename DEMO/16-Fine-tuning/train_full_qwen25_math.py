#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-Math-1.5B-Instruct 全量微调脚本
适用硬件：2× NVIDIA V100 16GB
精度方案：fp16 混合精度（V100 不支持 bf16，必须 fp16）+ DeepSpeed ZeRO-2
训练方式：full-parameter fine-tuning + gradient checkpointing + 小 batch + 梯度累积

数据格式：alpaca JSONL（支持 instruction/output，或 query/response）
示例行：
  {"instruction": "一个数的 3 倍加 5 等于 20，求这个数。",
   "output": "设这个数为 x。\n3x+5=20，解得 x=5。\n\n最终答案是 <<5>>",
   "meta_math_answer": "5"}

用法（2 卡）：
  torchrun --nproc_per_node=2 --master_port=29500 \
    train_full_qwen25_math.py \
    --model_name_or_path /Data/MODEL/Qwen/Qwen2.5-Math-1.5B-Instruct \
    --data_path data/metamath_40k.jsonl \
    --output_dir output/full-finetune \
    --deepspeed_config ds_config_zero2.json
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# Qwen2.5-Math-Instruct 的推理协议：逐步推理 + 最终答案包在 <<...>> 里
SYSTEM_PROMPT = (
    "You are a helpful assistant. Please reason step by step, "
    "and put the final answer inside <<...>>."
)


def build_messages(row):
    """把单条数据转成 chat 消息列表。"""
    query = row.get("instruction") or row.get("query", "")
    response = row.get("output") or row.get("response", "")
    # 脏数据行返回 None（转换脚本已过滤，这里兜底一层，避免单条坏数据中断训练）
    if not query.strip() or not response.strip() or len(response.strip()) < 20:
        return None

    # 若回答中没有 <<...>> 格式，用 meta_math_answer 字段补上，保持 Qwen2.5-Math 协议一致
    if "<<" not in response:
        ans = row.get("meta_math_answer") or row.get("answer")
        if ans:
            response = response.rstrip() + f"\n\n最终答案是 <<{ans}>>"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
        {"role": "assistant", "content": response},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
                        default="/Data/MODEL/Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--data_path", type=str, required=True,
                        help="alpaca 格式 JSONL 路径")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help="序列上限；OOM 时降到 768/512")
    parser.add_argument("--per_device_batch_size", type=int, default=1,
                        help="V100 16G 下全量微调只能 1，不要调大")
    parser.add_argument("--grad_accum", type=int, default=32,
                        help="梯度累积步数，有效 batch = 1×2卡×32 = 64")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="全量微调数学任务 1e-5 ~ 2e-5")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--deepspeed_config", type=str,
                        default="ds_config_zero2.json")
    parser.add_argument("--val_samples", type=int, default=1000,
                        help="从数据中切出的验证条数")
    args = parser.parse_args()

    assert os.path.isdir(args.model_name_or_path), \
        f"底模目录不存在: {args.model_name_or_path}"
    assert os.path.isfile(args.data_path), f"数据文件不存在: {args.data_path}"
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- 1. 加载模型 ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Qwen2.5 无 pad，用 eos 顶替

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,   # 关键：V100 用 fp16，不要用 bf16
    )
    model.config.use_cache = False   # 训练必须关 cache（配合梯度检查点）

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print(f"[INFO] GPU0 显存: 空闲 {free/1e9:.1f}GB / 总 {total/1e9:.1f}GB")

    # ---------- 2. 加载并预处理数据 ----------
    ds = load_dataset("json", data_files=args.data_path, split="train")
    test_n = min(args.val_samples, max(100, len(ds) // 10))
    split = ds.train_test_split(test_size=test_n, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"[INFO] 训练 {len(train_ds)} 条, 验证 {len(eval_ds)} 条")

    def preprocess(examples):
        texts = []
        for i in range(len(examples[list(examples.keys())[0]])):
            row = {k: examples[k][i] for k in examples.keys()}
            msgs = build_messages(row)
            if msgs is None:
                continue
            texts.append(
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
            )
        if not texts:
            raise ValueError("所有数据行均无效，请检查数据文件")
        enc = tokenizer(
            texts, truncation=True, max_length=args.max_seq_len, padding=False
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": [ids.copy() for ids in enc["input_ids"]],
        }

    train_ds = train_ds.map(preprocess, batched=True,
                            remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(preprocess, batched=True,
                          remove_columns=eval_ds.column_names)

    # ---------- 3. 训练参数 ----------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,             # 数学 SFT 一般不开 weight decay
        fp16=True,                    # V100 必选 fp16
        deepspeed=args.deepspeed_config,
        gradient_checkpointing=True,  # 省显存关键
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",   # transformers 4.40 用 evaluation_strategy（4.46+ 才叫 eval_strategy）
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",             # 有 wandb 可改成 "wandb"
        ddp_find_unused_parameters=False,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100
        ),
    )

    # ---------- 4. 训练 + 保存 ----------
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[DONE] 全量微调完成，模型已保存: {args.output_dir}")
    print(f"[NEXT] 部署: vllm serve {args.output_dir} --port 8000")


if __name__ == "__main__":
    main()
