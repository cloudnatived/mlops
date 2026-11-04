#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-Math-1.5B-Instruct LoRA 微调 + 推理 + 评测（最终可运行版）
适用：单卡/单机，V100 16G/24G
依赖：
  pip install torch transformers datasets peft trl sentencepiece
说明：
  - 支持 JSONL 数据文件
  - 也支持 Hugging Face 数据集路径（例如本地数据集目录）
  - 训练时使用 LoRA，显存更友好
  - 训练后自动做 3-5 个例子推理验证
"""

import argparse
import os
import json
import random
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

# ======================================================
# 基础工具
# ======================================================

def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()

def normalize_row(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    统一成：
      question: 问题
      answer: 答案/解题过程
    兼容多种字段名：
      instruction/query/problem/question/prompt
      output/response/solution/answer/generated_solution
    """
    question = (
        row.get("instruction")
        or row.get("query")
        or row.get("problem")
        or row.get("question")
        or row.get("prompt")
        or ""
    )

    answer = (
        row.get("output")
        or row.get("response")
        or row.get("solution")
        or row.get("answer")
        or row.get("generated_solution")
        or row.get("final_answer")
        or ""
    )

    q = safe_str(question)
    a = safe_str(answer)

    if not q or not a:
        return None

    # 过滤极短无效答案
    if len(a) < 10:
        return None

    return {"question": q, "answer": a}

def build_chat_text(row: Dict[str, Any], tokenizer: AutoTokenizer) -> Optional[str]:
    q = safe_str(row.get("question", ""))
    a = safe_str(row.get("answer", ""))

    if not q or not a:
        return None

    # 若答案里没有最终答案格式，就补一个保守版本
    if "<< " not in a and "<<" not in a and ">>" not in a:
        # 只补充，不强行改写原答案
        if len(a) > 8:
            a = a.rstrip() + "\n\n最终答案是 <<请检查并给出最终结果>>"

    messages = [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    if not text or len(text.strip()) < 10:
        return None

    return text

def preprocess_batch(examples: Dict[str, List[Any]], tokenizer: AutoTokenizer) -> Dict[str, List[str]]:
    """
    关键修正：
      - 必须保证输出长度和 batch 长度一致
      - 否则 datasets.map 在训练时会触发 IndexError
    """
    row_count = len(next(iter(examples.values())))
    texts = [""] * row_count

    for i in range(row_count):
        row = {k: examples[k][i] for k in examples.keys()}
        normalized = normalize_row(row)
        if normalized is None:
            texts[i] = ""
            continue

        text = build_chat_text(normalized, tokenizer)
        texts[i] = text if text is not None else ""

    return {"text": texts}

# ======================================================
# 数据加载
# ======================================================

def load_or_build_dataset(data_source: str):
    """
    支持:
      1) 本地 JSONL 文件: /path/to/data.jsonl
      2) Hugging Face 数据集路径: /Data/MODEL/meta-math/MetaMathQA
    """
    if data_source.lower().endswith(".jsonl"):
        ds = load_dataset("json", data_files=data_source, split="train")
    else:
        ds = load_dataset(data_source, split="train")

    if isinstance(ds, dict):
        # 如果数据集返回 dict，取 train
        ds = ds["train"]

    return ds

# ======================================================
# 训练逻辑
# ======================================================

def prepare_dataset_for_training(ds, tokenizer, val_samples: int, seed: int):
    # 先去掉明显坏掉的样本
    valid_rows = []
    for row in ds:
        normalized = normalize_row(row)
        if normalized is not None:
            valid_rows.append(normalized)

    if len(valid_rows) == 0:
        raise ValueError("数据集中没有有效样本，请检查字段名和内容。")

    ds = load_dataset("json", data_files=None) if False else None  # 仅占位，避免未用
    # 使用 list 形式构造新 dataset，方便控制清洗后的长度
    # 这里不依赖 datasets 的内部实现
    from datasets import Dataset

    clean_ds = Dataset.from_list(valid_rows)

    # 切分验证集
    val_size = min(val_samples, max(50, len(clean_ds) // 10))
    if val_size >= len(clean_ds):
        val_size = max(1, len(clean_ds) // 5)

    split = clean_ds.train_test_split(test_size=val_size, seed=seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    # 生成 text 列
    train_ds = train_ds.map(
        lambda batch: preprocess_batch(batch, tokenizer),
        batched=True,
        batch_size=256,
        desc="preprocess train"
    )
    eval_ds = eval_ds.map(
        lambda batch: preprocess_batch(batch, tokenizer),
        batched=True,
        batch_size=256,
        desc="preprocess eval"
    )

    # 过滤空 text
    train_ds = train_ds.filter(lambda x: len(x["text"].strip()) > 0)
    eval_ds = eval_ds.filter(lambda x: len(x["text"].strip()) > 0)

    if len(train_ds) == 0 or len(eval_ds) == 0:
        raise ValueError("训练或验证集为空，检查数据清洗逻辑。")

    print(f"[INFO] 训练集: {len(train_ds)} 条")
    print(f"[INFO] 验证集: {len(eval_ds)} 条")
    print("[INFO] 例子文本预览：")
    print(train_ds[0]["text"][:300])

    return train_ds, eval_ds

# ======================================================
# 推理与评测
# ======================================================

def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
    do_sample: bool = False
) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()

# ======================================================
# 主函数
# ======================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="JSONL 文件或 Hugging Face 本地数据集路径")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--eval_prompts", type=str, default=None, help="评测题目文件路径，JSON list 或 text file，每行一个题目")
    args = parser.parse_args()

    # 运行环境
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 1. 模型加载
    print(f"[INFO] 加载 tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] 加载模型: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model.config.use_cache = False

    # 2. 数据加载
    print(f"[INFO] 加载数据: {args.data_path}")
    raw_ds = load_or_build_dataset(args.data_path)
    train_ds, eval_ds = prepare_dataset_for_training(raw_ds, tokenizer, args.val_samples, args.seed)

    # 3. LoRA 配置
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # 4. 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        evaluation_strategy="epoch",
        metric_for_best_model="eval_loss",
        load_best_model_at_end=False,
        report_to="none",
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    # 5. 训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
        dataset_num_proc=1,
        tokenizer=tokenizer,
    )

    # 6. 训练
    print("[INFO] 开始训练...")
    trainer.train()

    # 7. 保存
    save_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[DONE] 训练完成，模型已保存到: {save_dir}")

    # 8. 推理验证
    print("[INFO] 开始推理评测...")
    eval_prompts = []

    if args.eval_prompts:
        if os.path.isfile(args.eval_prompts):
            ext = os.path.splitext(args.eval_prompts)[1].lower()
            with open(args.eval_prompts, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if ext == ".json":
                eval_prompts = json.loads(raw)
            else:
                eval_prompts = [line.strip() for line in raw.splitlines() if line.strip()]
        else:
            # 直接当成一个字符串列表 JSON
            try:
                eval_prompts = json.loads(args.eval_prompts)
            except Exception:
                eval_prompts = [args.eval_prompts]
    else:
        eval_prompts = [
            "已知 x^2 - 5x + 6 = 0，求 x 的值。",
            "一辆汽车每小时走 60 公里，3 小时后还剩 120 公里，求总路程。",
            "若 a = 7，b = 3，求 a^2 + 2ab + b^2 的值。",
            "一个数的 3 倍加 5 等于 20，求这个数。",
            "求方程 2x + 7 = 21 的解。"
        ]

    for idx, prompt in enumerate(eval_prompts[:10], start=1):
        print(f"\n======== 评测题 {idx} ========")
        print(prompt)
        result = generate_answer(model, tokenizer, prompt, max_new_tokens=256, do_sample=False)
        print(result)

    print("\n[INFO] 评测结束。")


if __name__ == "__main__":
    main()
