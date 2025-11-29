#!/usr/bin/env python3
import torch
import deepspeed
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    
    # 简单模型
    model = torch.nn.Linear(10, 10)
    
    # 最小配置
    ds_config = {
        "train_batch_size": 4,
        "train_micro_batch_size_per_gpu": 2,
        "optimizer": {"type": "Adam", "params": {"lr": 1e-3}},
        "zero_optimization": {"stage": 1}
    }
    
    # 初始化DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=model.parameters()
    )
    
    # 简单训练步骤
    data = torch.randn(2, 10).cuda()
    target = torch.randn(2, 10).cuda()
    
    output = model_engine(data)
    loss = torch.nn.functional.mse_loss(output, target)
    
    model_engine.backward(loss)
    model_engine.step()
    
    print("✅ DeepSpeed初始化成功！")

if __name__ == "__main__":
    main()
