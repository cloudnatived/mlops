#!/bin/bash

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 创建输出目录
mkdir -p outputs

echo "=========================================="
echo "🧪 开始微调测试实验"
echo "=========================================="

# 实验1: Gemma-2-2B 全量微调（挑战显存极限）
echo ""
echo "📊 实验1: 全量微调 - Gemma-2-2B"
echo "------------------------------------------"
python scripts/train_full.py \
    --model_name google/gemma-2-2b \
    --data_path ./data/sample_data.jsonl \
    --output_dir ./outputs/exp1_full_2b \
    --num_epochs 2 \
    --batch_size 1 \
    --grad_accum 16 \
    --learning_rate 2e-5 \
    --max_length 512

# 实验2: Gemma-2-2B LoRA（仅Q/V）
echo ""
echo "📊 实验2: LoRA(Q/V) - Gemma-2-2B"
echo "------------------------------------------"
python scripts/train_lora.py \
    --model_name google/gemma-2-2b \
    --data_path ./data/sample_data.jsonl \
    --output_dir ./outputs/exp2_lora_qv_2b \
    --num_epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --learning_rate 2e-4 \
    --max_length 512 \
    --lora_r 8 \
    --lora_alpha 16

# 实验3: Gemma-2-2B LoRA（全部线性层）
echo ""
echo "📊 实验3: LoRA(All) - Gemma-2-2B"
echo "------------------------------------------"
python scripts/train_lora.py \
    --model_name google/gemma-2-2b \
    --data_path ./data/sample_data.jsonl \
    --output_dir ./outputs/exp3_lora_all_2b \
    --num_epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --learning_rate 2e-4 \
    --max_length 512 \
    --lora_r 8 \
    --lora_alpha 16 \
    --target_all

# 实验4: Gemma-2-9B LoRA（全部线性层）
echo ""
echo "📊 实验4: LoRA(All) - Gemma-2-9B"
echo "------------------------------------------"
python scripts/train_lora.py \
    --model_name google/gemma-2-9b \
    --data_path ./data/sample_data.jsonl \
    --output_dir ./outputs/exp4_lora_all_9b \
    --num_epochs 3 \
    --batch_size 2 \
    --grad_accum 16 \
    --learning_rate 2e-4 \
    --max_length 512 \
    --lora_r 16 \
    --lora_alpha 32 \
    --target_all

# 实验5: Gemma-2-9B QLoRA（4-bit量化）
echo ""
echo "📊 实验5: QLoRA - Gemma-2-9B"
echo "------------------------------------------"
python scripts/train_qlora.py \
    --model_name google/gemma-2-9b \
    --data_path ./data/sample_data.jsonl \
    --output_dir ./outputs/exp5_qlora_9b \
    --num_epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --learning_rate 2e-4 \
    --max_length 512 \
    --lora_r 16 \
    --lora_alpha 32 \
    --target_all

# 实验6: Gemma-2-27B QLoRA（大模型挑战）
echo ""
echo "📊 实验6: QLoRA - Gemma-2-27B (CPU Offload)"
echo "------------------------------------------"
python scripts/train_qlora.py \
    --model_name google/gemma-2-27b \
    --data_path ./data/sample_data.jsonl \
    --output_dir ./outputs/exp6_qlora_27b \
    --num_epochs 2 \
    --batch_size 1 \
    --grad_accum 32 \
    --learning_rate 2e-4 \
    --max_length 512 \
    --lora_r 16 \
    --lora_alpha 32 \
    --target_all \
    --use_cpu_offload

# 实验7: Gemma-2-2B P-Tuning
echo ""
echo "📊 实验7: P-Tuning - Gemma-2-2B"
echo "------------------------------------------"
python scripts/train_ptuning.py \
    --model_name google/gemma-2-2b \
    --data_path ./data/sample_data.jsonl \
    --output_dir ./outputs/exp7_ptuning_2b \
    --num_epochs 5 \
    --batch_size 8 \
    --grad_accum 4 \
    --learning_rate 1e-4 \
    --max_length 512 \
    --num_virtual_tokens 20

echo ""
echo "=========================================="
echo "✅ 所有实验完成！"
echo "=========================================="
echo ""
echo "📁 实验结果保存在: ./outputs/"
echo ""
echo "💡 查看显存和训练日志:"
echo "  - 每个实验目录下有训练日志"
echo "  - 使用 nvidia-smi 监控显存使用"
