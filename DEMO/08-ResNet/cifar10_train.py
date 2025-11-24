#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# cifar10_train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet152, ResNet152_Weights
from tqdm import tqdm
import time
import argparse
import logging
from datetime import timedelta

# TensorBoard 支持
from torch.utils.tensorboard import SummaryWriter

# 从 utils 导入数据相关
from cifar10_utils import setup_distributed, cleanup, get_dataloaders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, help='Rank of this process (0~5)')
    parser.add_argument('--world_size', type=int, default=6, help='Total number of GPUs (3 nodes × 2)')
    parser.add_argument('--master_addr', type=str, default='172.18.8.200', help='IP of node 0')
    parser.add_argument('--epochs', type=int, default=100, help='Total epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Per-GPU batch size')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--data_dir', type=str, default='/tmp/cifar10', help='Dataset path')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint save directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='TensorBoard log directory')
    parser.add_argument('--port', type=str, default='29500', help='Port for distributed training')
    args = parser.parse_args()

    # 分布式初始化
    setup_distributed(args.rank, args.world_size, args.master_addr, args.port)
    device = torch.device(f'cuda:{torch.cuda.current_device()}')

    # 仅在 rank 0 上创建目录和日志
    if args.rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        # 初始化 TensorBoard writer
        writer = SummaryWriter(log_dir=args.log_dir)
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(args.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Starting training: {args.world_size} GPUs, {args.epochs} epochs")
    else:
        writer = None
        logger = None

    try:
        # ========== 模型构建 ==========
        model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 10 classes
        model = model.to(device)
        model = DDP(model, device_ids=[device.index], output_device=device.index)

        # ========== 数据加载 ==========
        train_loader, test_loader, train_sampler = get_dataloaders(args.batch_size, args.data_dir)
        total_steps_per_epoch = len(train_loader)

        # ========== 优化器 & 学习率调度 ==========
        # 分层学习率：FC 层高 LR，主干低 LR
        optimizer = optim.SGD([
            {'params': model.module.fc.parameters(), 'lr': 0.01},
            {'params': [p for n, p in model.module.named_parameters() if 'fc' not in n], 'lr': 1e-4}  # 主干
        ], momentum=0.9, weight_decay=5e-4)

        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        # ========== 训练状态变量 ==========
        best_acc = 0.0
        start_time = time.time()

        # ========== 训练循环 ==========
        for epoch in range(args.epochs):
            model.train()
            train_sampler.set_epoch(epoch)
            running_loss = 0.0
            correct = 0
            total = 0

            optimizer.zero_grad()

            # 使用 tqdm 显示进度（仅 rank 0）
            data_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", 
                            disable=args.rank != 0, leave=False)

            for step, (inputs, targets) in enumerate(data_iter):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets) / args.accumulation_steps
                loss.backward()

                running_loss += loss.item() * args.accumulation_steps
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # 梯度累积
                if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            # ========== 同步训练指标 ==========
            # 收集所有进程的训练损失和准确率
            train_loss_tensor = torch.tensor([running_loss / len(train_loader)], device=device)
            train_acc_tensor = torch.tensor([100. * correct / total], device=device)
            
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.SUM)
            
            avg_train_loss = train_loss_tensor.item() / dist.get_world_size()
            train_acc = train_acc_tensor.item() / dist.get_world_size()

            # ========== 验证 ==========
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            # 同步验证指标
            val_loss_tensor = torch.tensor([val_loss / len(test_loader)], device=device)
            val_acc_tensor = torch.tensor([100. * val_correct / val_total], device=device)
            
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.SUM)
            
            avg_val_loss = val_loss_tensor.item() / dist.get_world_size()
            val_acc = val_acc_tensor.item() / dist.get_world_size()

            # ========== 更新学习率 ==========
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']  # FC 层 LR

            # ========== 仅在 rank 0 进行日志和保存 ==========
            if args.rank == 0:
                # ========== 计算时间 ==========
                elapsed_time = time.time() - start_time
                estimated_total = elapsed_time / (epoch + 1) * args.epochs
                eta = estimated_total - elapsed_time

                # ========== 保存模型 ==========
                checkpoint_path = os.path.join(args.save_dir, f"epoch_{epoch+1:03d}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_acc': val_acc,
                    'elapsed_time': elapsed_time,
                }, checkpoint_path)

                # 保存最佳模型
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_path = os.path.join(args.save_dir, "best_model.pth")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.module.state_dict(),
                        'val_acc': val_acc,
                    }, best_path)
                    logger.info(f"🎉🎉 Best model saved with accuracy: {best_acc:.2f}%")

                # ========== TensorBoard 日志 ==========
                writer.add_scalar('Loss/Train', avg_train_loss, epoch)
                writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
                writer.add_scalar('Accuracy/Train', train_acc, epoch)
                writer.add_scalar('Accuracy/Validation', val_acc, epoch)
                writer.add_scalar('Learning Rate/FC', current_lr, epoch)
                writer.add_scalar('Learning Rate/Backbone', optimizer.param_groups[1]['lr'], epoch)

                # ========== 日志输出 ==========
                logger.info(
                    f"Epoch {epoch+1:03d}/{args.epochs} | "
                    f"Train Loss: {avg_train_loss:.3f}, Acc: {train_acc:.2f}% | "
                    f"Val Loss: {avg_val_loss:.3f}, Acc: {val_acc:.2f}% | "
                    f"LR: {current_lr:.1e} | "
                    f"Time: {timedelta(seconds=int(elapsed_time))} | "
                    f"ETA: {timedelta(seconds=int(eta))}"
                )

        # ========== 训练结束 ==========
        if args.rank == 0:
            total_time = time.time() - start_time
            logger.info(f"✅ Training completed in {timedelta(seconds=int(total_time))}.")
            logger.info(f"📈📈 Best validation accuracy: {best_acc:.2f}%")
            writer.close()

    except Exception as e:
        if args.rank == 0:
            logger.error(f"Training failed with error: {str(e)}")
        raise e
    finally:
        cleanup()

if __name__ == '__main__':
    main()
