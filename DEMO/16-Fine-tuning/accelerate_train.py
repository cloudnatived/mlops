import argparse
import os

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from bitsandbytes.optim import AdamW8bit
from datasets import interleave_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)


DATASETS = [
    "/Data/MODEL/meta-math/MetaMathQA",
    "/Data/MODEL/nvidia/OpenMathInstruct-2",
    "/Data/MODEL/AI-MO/NuminaMath-CoT",
    "/Data/MODEL/open-r1/OpenR1-Math-220k",
]

PROBLEM_KEYS = (
    "problem",
    "query",
    "question",
    "prompt",
    "instruction",
)

ANSWER_KEYS = (
    "generated_solution",
    "solution",
    "response",
    "reasoning",
    "rationale",
    "answer",
    "completion",
)


def nonempty_text(value):
    """把数据集字段转换为非空字符串；忽略缺失或非文本值。"""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, (int, float)):
        return str(value)
    return None


def extract_pair(example):
    """兼容常见数学数据集字段，以及 messages 格式。"""
    messages = example.get("messages")
    if isinstance(messages, list):
        user_text = None
        assistant_text = None
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            content = nonempty_text(message.get("content"))
            if not content:
                continue
            if role in ("user", "human"):
                user_text = content
            elif role in ("assistant", "gpt"):
                assistant_text = content
        if user_text and assistant_text:
            return user_text, assistant_text

    problem = next(
        (text for key in PROBLEM_KEYS if (text := nonempty_text(example.get(key)))),
        None,
    )
    answer = next(
        (text for key in ANSWER_KEYS if (text := nonempty_text(example.get(key)))),
        None,
    )
    if problem and answer:
        return problem, answer

    return None


def has_pair(example):
    return extract_pair(example) is not None


def tokenize_example(example, tokenizer, max_length):
    pair = extract_pair(example)
    if pair is None:
        return {"input_ids": [], "labels": []}

    problem, answer = pair
    messages = [
        {
            "role": "user",
            "content": (
                "Solve the following math problem. Show the reasoning "
                "and provide the final answer.\n\n"
                f"{problem}"
            ),
        }
    ]

    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": answer}],
        tokenize=True,
        add_generation_prompt=False,
    )

    # 找到 prompt 与完整对话共同的前缀，避免模板差异导致标签错位。
    prefix_len = 0
    for prompt_token, full_token in zip(prompt_ids, full_ids):
        if prompt_token != full_token:
            break
        prefix_len += 1

    answer_ids = full_ids[prefix_len:]
    prompt_prefix_ids = full_ids[:prefix_len]

    # 超长时优先保留答案，再保留尽可能多的 prompt 尾部。
    if len(answer_ids) >= max_length:
        answer_ids = answer_ids[:max_length]
        prompt_prefix_ids = []
    else:
        prompt_budget = max_length - len(answer_ids)
        prompt_prefix_ids = prompt_prefix_ids[-prompt_budget:]

    input_ids = prompt_prefix_ids + answer_ids
    labels = [-100] * len(prompt_prefix_ids) + answer_ids

    return {"input_ids": input_ids, "labels": labels}


class CausalLMCollator:
    """动态 padding；labels 的 padding 值设为 -100。"""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        features = [f for f in features if f["input_ids"] and f["labels"]]
        if not features:
            raise ValueError("当前 batch 没有可用于训练的样本。")

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []

        for feature in features:
            ids = feature["input_ids"]
            target = feature["labels"]
            pad_len = max_len - len(ids)

            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(target + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_dataset(tokenizer, max_length, seed, shuffle_buffer):
    streams = []

    for dataset_name in DATASETS:
        print(f"Loading streaming dataset: {dataset_name}")
        stream = load_dataset(
            dataset_name,
            split="train",
            streaming=True,
        )
        stream = stream.filter(has_pair)
        stream = stream.map(
            lambda example: tokenize_example(example, tokenizer, max_length)
        )
        stream = stream.filter(lambda example: len(example["labels"]) > 0)
        streams.append(stream)

    # 四个数据源按相同概率交错采样；流式读取避免把全部数据集装入内存。
    combined = interleave_datasets(
        streams,
        probabilities=[0.25, 0.25, 0.25, 0.25],
        seed=seed,
        stopping_strategy="all_exhausted",
    )
    return combined.shuffle(buffer_size=shuffle_buffer, seed=seed)


def save_model(accelerator, model, tokenizer, output_dir):
    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)

    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
        safe_serialization=True,
    )

    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        print(f"模型已保存至：{output_dir}")

    accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
    )
    parser.add_argument("--output_dir", default="./accelerate-qwen-math-full-ft")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--shuffle_buffer", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    #ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False,gradient_as_bucket_view=True,)
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        #dispatch_batches=False,
        #split_batches=False,
        kwargs_handlers=[ddp_kwargs],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        #torch_dtype=torch.float16,
        torch_dtype=torch.float32,    # ← 关键改动：FP32 加载
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # 参数没有冻结：这是全量微调。
    #optimizer = AdamW8bit(    # 参数没有冻结：这是全量微调。
    optimizer = torch.optim.AdamW(    #改用标准 PyTorch AdamW，避免与 accelerate 的 GradScaler 冲突
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_dataset = build_dataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        collate_fn=CausalLMCollator(tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=True,
    )

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"每卡 batch size: {args.per_device_batch_size}")
        print(
            "有效 batch size: "
            f"{args.per_device_batch_size} × "
            f"{accelerator.num_processes} 卡 × "
            f"{args.gradient_accumulation_steps} 梯度累积"
        )
        print(f"训练步数: {args.max_steps}")

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    loss_window = []

    # IterableDataset 没有固定长度；数据流耗尽后重新迭代，直到达到 max_steps。
    while global_step < args.max_steps:
        saw_batch = False

        for batch in train_dataloader:
            saw_batch = True

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    pass   # 暂时关闭梯度裁剪，测试用
                    #accelerator.clip_grad_norm_(model.parameters(), 1.0)    #触发这次崩溃的直接调用点。全量微调初期梯度可能不稳

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                loss_window.append(loss.detach().float())

                if (
                    global_step % args.logging_steps == 0
                    and accelerator.is_main_process
                ):
                    mean_loss = torch.stack(loss_window).mean().item()
                    current_lr = lr_scheduler.get_last_lr()[0]
                    print(
                        f"step={global_step}/{args.max_steps} "
                        f"loss={mean_loss:.4f} lr={current_lr:.2e}"
                    )
                    loss_window.clear()

                if global_step >= args.max_steps:
                    break

        if not saw_batch:
            raise RuntimeError(
                "数据流没有产出样本。请确认数据集有 train split，"
                "且字段映射能找到题目和答案。"
            )

    save_model(accelerator, model, tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
