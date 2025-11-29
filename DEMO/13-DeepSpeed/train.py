#!/usr/bin/env python3
import os
import torch
import deepspeed
import argparse
import json
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.simple_transformer import create_model
from data.dummy_dataset import create_data_loader

def setup_training():
    """设置训练参数"""
    parser = argparse.ArgumentParser(description='DeepSpeed Demo Training')
    
    # 模型参数
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--max_seq_length', type=int, default=1024)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=1000)
    
    # DeepSpeed参数
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--deepspeed_config', type=str, default='configs/ds_config.json')
    
    # 分布式参数
    parser.add_argument('--master_addr', type=str, default='localhost')
    parser.add_argument('--master_port', type=str, default='29500')
    
    return parser.parse_args()

def initialize_distributed(args):
    """初始化分布式训练"""
    torch.cuda.set_device(args.local_rank)
    deepspeed.init_distributed()
    
    args.world_size = torch.distributed.get_world_size()
    args.global_rank = torch.distributed.get_rank()
    
    print(f"🚀 初始化分布式训练: rank {args.global_rank}/{args.world_size}")
    
    # 设置主节点地址和端口
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

def train_epoch(model, dataloader, sampler, epoch, global_step, writer, args):
    """训练一个epoch"""
    model.train()
    sampler.set_epoch(epoch)
    
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=args.global_rank != 0)
    
    for batch_idx, batch in enumerate(progress_bar):
        # 将数据移动到GPU
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['labels'].cuda()
        
        # 前向传播和损失计算
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        
        # 反向传播和优化
        model.backward(loss)
        model.step()
        
        # 记录损失
        total_loss += loss.item()
        
        # 记录到TensorBoard（只在rank 0上记录）
        if args.global_rank == 0 and global_step % 10 == 0:
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/learning_rate', 
                             model.get_lr()[0] if hasattr(model, 'get_lr') else args.learning_rate, 
                             global_step)
        
        # 打印训练信息
        if global_step % 100 == 0 and args.global_rank == 0:
            print(f'Step {global_step}: Loss = {loss.item():.4f}')
        
        # 保存检查点
        if global_step % args.save_interval == 0 and args.global_rank == 0:
            save_checkpoint(model, global_step, args)
        
        global_step += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss, global_step

def save_checkpoint(model, step, args):
    """保存检查点"""
    checkpoint_dir = f'checkpoints/step_{step}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.save_checkpoint(checkpoint_dir)
    print(f"✅ 检查点已保存: {checkpoint_dir}")

def evaluate_model(model, dataloader, args):
    """评估模型"""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', disable=args.global_rank != 0):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    
    # 在所有rank上同步损失
    if args.world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss).cuda()
        torch.distributed.all_reduce(avg_loss_tensor)
        avg_loss = avg_loss_tensor.item() / args.world_size
    
    return avg_loss

def main():
    # 设置参数
    args = setup_training()
    
    # 初始化分布式训练
    initialize_distributed(args)
    
    # 只在rank 0上创建TensorBoard writer
    if args.global_rank == 0:
        writer = SummaryWriter('runs/deepspeed_demo')
        print("📊 TensorBoard日志目录: runs/deepspeed_demo")
    else:
        writer = None
    
    # 创建模型
    model = create_model(args)
    
    # 加载DeepSpeed配置
    with open(args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)
    
    # 初始化DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    # 创建数据加载器
    dataloader, sampler = create_data_loader(args, args.global_rank, args.world_size)
    
    print(f"🎯 开始训练: {args.world_size}个GPU, {len(dataloader)}个批次/epoch")
    
    # 训练循环
    global_step = 0
    for epoch in range(args.epochs):
        if args.global_rank == 0:
            print(f"\n📈 Epoch {epoch+1}/{args.epochs}")
        
        # 训练一个epoch
        start_time = time.time()
        avg_loss, global_step = train_epoch(
            model_engine, dataloader, sampler, epoch, global_step, writer, args
        )
        epoch_time = time.time() - start_time
        
        # 评估模型
        eval_loss = evaluate_model(model_engine, dataloader, args)
        
        # 打印训练结果（只在rank 0上）
        if args.global_rank == 0:
            print(f"✅ Epoch {epoch+1} 完成:")
            print(f"   训练损失: {avg_loss:.4f}")
            print(f"   评估损失: {eval_loss:.4f}")
            print(f"   时间: {epoch_time:.2f}秒")
            print(f"   全局步数: {global_step}")
            
            # 记录到TensorBoard
            if writer:
                writer.add_scalar('train/epoch_loss', avg_loss, epoch)
                writer.add_scalar('eval/loss', eval_loss, epoch)
                writer.add_scalar('train/epoch_time', epoch_time, epoch)
    
    # 保存最终模型
    if args.global_rank == 0:
        save_checkpoint(model_engine, global_step, args)
        print("🎉 训练完成！")
    
    # 清理
    if writer:
        writer.close()

if __name__ == "__main__":
    main()
