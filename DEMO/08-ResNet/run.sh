#!/bin/bash
# run.sh

MASTER_ADDR="172.18.8.208" # 在MASTER_ADDR的rank0上创建目录和日志，只在rank0进程上打印汇总结果，每个epoch结束时保存检查点（只在rank0上执行）， # 训练结束后保存最终模型（只在rank0上执行）
MASTER_PORT="29500"
WORLD_SIZE=6
GPU_PER_NODE=2
RANK=$1

# 可选：设置 NCCL 参数优化 10G 网络    
export NCCL_SOCKET_IFNAME=enp4s1
export NCCL_DEBUG=INFO

cd /Data/DEMO/CODE/RESNET/;
python3 cifar10_train.py \
  --rank $RANK \
  --world_size $WORLD_SIZE \
  --master_addr $MASTER_ADDR \
  --epochs 100 \
  --batch_size 32 \
  --accumulation_steps 4 \
  --data_dir ./data \
  --save_dir ./checkpoints \
  --log_dir ./logs
