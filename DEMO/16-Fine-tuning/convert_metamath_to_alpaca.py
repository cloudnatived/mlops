#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MetaMathQA-395K.json → LLaMA-Factory/自定义训练脚本用的 alpaca JSONL
输出: {"instruction": 题目, "output": 带 CoT 的回答(已补 <<...>> 最终答案), "meta_math_answer": 答案}

用法:
  python3 convert_metamath_to_alpaca.py \
    --src /Data/MODEL/meta-math/MetaMathQA/MetaMathQA-395K.json \
    --dst /Data/DEMO/Qwen2.5-Math-1.5B-Instruct/data/metamath_40k.jsonl \
    --num 40000 --seed 42
"""

import argparse
import json
import os
import random
import re

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

def normalize_response(item):
    """保证回答带 Qwen2.5-Math 的 <<...>> 最终答案格式，语言与回答正文一致。"""
    response = item.get("response") or item.get("output") or ""
    ans = item.get("meta_math_answer") or item.get("answer")
    if ans and "<<" not in response:
        marker = "最终答案是" if _CJK_RE.search(response) else "The final answer is"
        response = response.rstrip() + f"\n\n{marker} <<{ans}>>"
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="MetaMathQA-395K.json 路径")
    parser.add_argument("--dst", type=str, required=True, help="输出 jsonl 路径")
    parser.add_argument("--num", type=int, default=40000, help="采样条数（实验建议 20K~40K）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[1/3] 加载 {args.src} ...")
    with open(args.src) as f:
        data = json.load(f)

    # 兼容 {"data": [...]} 或直接 [...] 两种结构
    if isinstance(data, dict):
        data = data.get("data") or list(data.values())[0]
    print(f"[2/3] 总条数: {len(data)}")
    print(f"      字段: {list(data[0].keys())}")
    print(f"      示例: {json.dumps(data[0], ensure_ascii=False)[:200]}")

    # 过滤脏数据：空题目、空/过短回答（MetaMathQA-395K.json 内含少量坏行）
    def is_valid(item):
        query = (item.get("query") or item.get("instruction") or "").strip()
        response = (item.get("response") or item.get("output") or "").strip()
        return len(query) >= 3 and len(response) >= 20

    valid = [it for it in data if is_valid(it)]
    dropped = len(data) - len(valid)
    print(f"[2.5] 过滤无效行 {dropped} 条，剩余 {len(valid)} 条")
    if not valid:
        raise SystemExit("有效数据为 0，无法继续")

    random.seed(args.seed)
    random.shuffle(valid)
    sample = valid[: args.num]
    print(f"[3/3] 采样 {len(sample)} 条，写出 {args.dst} ...")

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    with open(args.dst, "w") as f:
        for item in sample:
            f.write(json.dumps({
                "instruction": item.get("query") or item.get("instruction", ""),
                "output": normalize_response(item),
                "meta_math_answer": item.get("meta_math_answer") or "",
            }, ensure_ascii=False) + "\n")

    # 打印 3 条抽样做人工检查
    print("\n===== 抽样检查（前 3 条）=====")
    with open(args.dst) as f:
        for i, line in zip(range(3), f):
            r = json.loads(line)
            print(f"--- 第{i+1}条 ---")
            print("Q:", r["instruction"][:120])
            print("A:", r["output"][-200:])
            print()
    print("完成。可直接用于训练: --data_path", args.dst)

if __name__ == "__main__":
    main()
