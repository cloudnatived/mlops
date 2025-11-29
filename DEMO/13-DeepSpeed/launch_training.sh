#!/bin/bash
# 分布式训练启动脚本

# 集群节点配置
MASTER_ADDR="172.18.8.208"  # 主节点IP
MASTER_PORT="29500"
NUM_NODES=3
GPUS_PER_NODE=2

# 训练配置
CONFIG="configs/ds_config.json"  # 或 ds_config_zero3.json
EPOCHS=10
BATCH_SIZE=8
HIDDEN_SIZE=768
NUM_LAYERS=12

# 创建日志目录
mkdir -p logs
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

echo "🚀 启动DeepSpeed分布式训练"
echo "📊 集群配置: $NUM_NODES节点 × $GPUS_PER_NODE GPU"
echo "🎯 主节点: $MASTER_ADDR:$MASTER_PORT"
echo "📁 配置文件: $CONFIG"

# 使用deepspeed启动训练
deepspeed \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --num_nodes $NUM_NODES \
    --num_gpus $GPUS_PER_NODE \
    --no_local_rank \
    --no_python \
    train.py \
    --deepspeed_config $CONFIG \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --num_layers $NUM_LAYERS \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    2>&1 | tee $LOG_FILE

echo "✅ 训练完成！日志文件: $LOG_FILE"
