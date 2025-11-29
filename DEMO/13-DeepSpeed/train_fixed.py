#!/usr/bin/env python3
import os
import torch
import deepspeed
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def setup_training():
    """设置训练参数"""
    parser = argparse.ArgumentParser(description='DeepSpeed Training Demo')
    
    # 基础参数
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--fp16', action='store_true', help='启用FP16训练')
    
    # DeepSpeed参数
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--deepspeed_config', type=str, default='',
                       help='DeepSpeed配置文件路径')
    
    return parser.parse_args()

class SimpleModel(torch.nn.Module):
    def __init__(self, input_size=1000, hidden_size=500, output_size=10):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 自动类型转换
        if next(self.parameters()).dtype != x.dtype:
            x = x.to(next(self.parameters()).dtype)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def create_dummy_dataloader(args):
    """创建虚拟数据加载器"""
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=1000, input_size=1000, output_size=10):
            self.num_samples = num_samples
            self.input_size = input_size
            self.output_size = output_size
            
            # 根据是否启用FP16生成对应类型的数据
            if args.fp16:
                self.data = torch.randn(num_samples, input_size, dtype=torch.float16)
            else:
                self.data = torch.randn(num_samples, input_size)
                
            self.labels = torch.randint(0, output_size, (num_samples,))
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    
    dataset = DummyDataset(num_samples=args.num_samples)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    return dataloader

def initialize_deepspeed(model, args):
    """初始化DeepSpeed"""
    if args.deepspeed_config and os.path.exists(args.deepspeed_config):
        print(f"📁 使用DeepSpeed配置文件: {args.deepspeed_config}")
        model_engine, _, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=model.parameters()
        )
    else:
        print("🔧 使用内置DeepSpeed配置")
        ds_config = {
            "train_batch_size": args.batch_size,
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": args.learning_rate,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            "fp16": {
                "enabled": args.fp16
            },
            "zero_optimization": {
                "stage": 1
            }
        }
        model_engine, _, _, _ = deepspeed.initialize(
            config=ds_config,
            model=model,
            model_parameters=model.parameters()
        )
    
    return model_engine

def main():
    args = setup_training()
    
    # 初始化分布式
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        deepspeed.init_distributed()
        args.world_size = torch.distributed.get_world_size()
    else:
        args.world_size = 1
    
    print(f"🚀 开始训练: {args.world_size}个GPU, FP16={'启用' if args.fp16 else '禁用'}")
    
    # 创建模型和数据
    model = SimpleModel()
    dataloader = create_dummy_dataloader(args)
    
    # 初始化DeepSpeed
    model_engine = initialize_deepspeed(model, args)
    
    # 训练循环
    for epoch in range(args.epochs):
        model_engine.train()
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data = data.cuda()
            targets = targets.cuda()
            
            # 前向传播
            outputs = model_engine(data)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            
            # 反向传播
            model_engine.backward(loss)
            model_engine.step()
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    print("🎉 训练完成！")

if __name__ == "__main__":
    main()
