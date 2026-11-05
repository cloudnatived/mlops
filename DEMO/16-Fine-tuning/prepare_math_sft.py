import json
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import os

def process_metamath(example):
    # MetaMathQA 通常有 query / response 或类似字段
    q = example.get("query") or example.get("question") or example.get("problem", "")
    a = example.get("response") or example.get("answer") or example.get("solution", "")
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}

def process_openmath(example):
    # nvidia/OpenMathInstruct-2: problem + generated_solution
    return {
        "messages": [
            {"role": "user", "content": example["problem"]},
            {"role": "assistant", "content": example["generated_solution"]}
        ]
    }

def process_numina(example):
    # AI-MO/NuminaMath-CoT: problem + solution
    return {
        "messages": [
            {"role": "user", "content": example.get("problem", example.get("question", ""))},
            {"role": "assistant", "content": example.get("solution", example.get("answer", ""))}
        ]
    }

def process_openr1(example):
    # open-r1/OpenR1-Math-220k 通常已有 messages 或 problem + generations
    if "messages" in example:
        return {"messages": example["messages"]}
    # 兜底
    problem = example.get("problem", "")
    # 取正确的 generation（数据集有多个 traces，优先正确的）
    solution = example.get("solution") or example.get("generations", [""])[0]
    return {"messages": [{"role": "user", "content": problem}, {"role": "assistant", "content": solution}]}

def main():
    os.makedirs("data/math_sft", exist_ok=True)
    
    # 按需加载（注意内存，32GB 系统内存下建议分批或只用子集）

    # ds1: MetaMathQA
    ds1 = load_dataset("/Data/MODEL/meta-math/MetaMathQA", split="train")
    ds1 = ds1.map(process_metamath, remove_columns=ds1.column_names)

    # ds2: OpenMathInstruct-2 (1M 子集)
    ds2 = load_dataset("/Data/MODEL/nvidia/OpenMathInstruct-2", split="train_1M")
    ds2 = ds2.map(process_openmath, remove_columns=ds2.column_names)

    # ds3: NuminaMath-CoT
    ds3 = load_dataset("/Data/MODEL/AI-MO/NuminaMath-CoT", split="train")
    ds3 = ds3.map(process_numina, remove_columns=ds3.column_names)

    # ds4: OpenR1-Math-220k
    ds4 = load_dataset("/Data/MODEL/open-r1/OpenR1-Math-220k", split="train")
    ds4 = ds4.map(process_openr1, remove_columns=ds4.column_names)

    combined = concatenate_datasets([ds1, ds2, ds3, ds4])
    combined = combined.filter(lambda x: len(str(x["messages"])) < 12000)

    with open("data/math_sft/train.jsonl", "w", encoding="utf-8") as f:
        for ex in tqdm(combined):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")    
 
    # 合并 + 过滤过长样本
    combined = concatenate_datasets([ds1, ds2, ds3, ds4])
    # 简单长度过滤（用 tokenizer 更准）
    combined = combined.filter(lambda x: len(str(x["messages"])) < 12000)  # 粗过滤
    
    # 保存为 jsonl
    with open("data/math_sft/train.jsonl", "w", encoding="utf-8") as f:
        for ex in tqdm(combined):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print("Done. 建议再随机打乱并切分 1%~2% 做 validation。")

if __name__ == "__main__":
    main()
