#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertConfig, BertModel, BertTokenizer
from datetime import datetime
import argparse
import yaml
import logging
from tqdm import tqdm
import socket
import warnings

# 设置HF镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

warnings.filterwarnings("ignore")

# 创建全局 logger
# 创建 logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 清除可能已存在的 handler（避免重复）
if logger.hasHandlers():
    logger.handlers.clear()

# 创建 handler
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

# 使用不含 rank 的格式（我们通过 filter 注入到 message）
formatter = logging.Formatter(
    '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)

class RankFilter(logging.Filter):
    def filter(self, record):
        rank = int(os.environ.get("RANK", 0))
        record.msg = f"[Rank {rank}] {record.msg}"
        return True

handler.addFilter(RankFilter())
logger.addHandler(handler)

# 防止向上层 logger 传播（避免重复日志）
logger.propagate = False

def get_local_ip():
    """获取本地IP地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def setup_distributed():
    """初始化分布式训练环境"""
    os.environ['NCCL_DEBUG'] = 'INFO'
    #os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
    os.environ['NCCL_SOCKET_IFNAME'] = 'enp4s1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    
    # 使用环境变量设置 MASTER_ADDR 和 MASTER_PORT
    master_addr = os.environ.get('MASTER_ADDR', '172.18.8.208')
    master_port = os.environ.get('MASTER_PORT', '29500')
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank == -1:
        raise ValueError("LOCAL_RANK not set")
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    #logger.info(f"[Rank {rank}] Initialized with MASTER_ADDR={master_addr}, MASTER_PORT={master_port}, world_size={world_size}")
    logger.info(f"Initializing distributed training with: MASTER_ADDR={master_addr}, MASTER_PORT={master_port}, world_size={world_size}, rank={rank}, local_rank={local_rank}")
    return local_rank

def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def setup_logging(rank, log_dir):
    """设置日志"""
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    return logging.getLogger(__name__)

def load_config(config_path):
    """安全加载配置并转换数值类型"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'training' in config:
        training = config['training']
        for key in ['learning_rate', 'weight_decay']:
            if key in training:
                training[key] = float(training[key])
        for key in ['batch_size', 'accumulation_steps', 'epochs']:
            if key in training:
                training[key] = int(training[key])
    
    if 'data' in config and 'num_workers' in config['data']:
        config['data']['num_workers'] = int(config['data']['num_workers'])
    
    return config

class BertForPretraining(nn.Module):
    """BERT预训练模型，支持MLM和NSP"""
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.cls_mlm = nn.Linear(config.hidden_size, config.vocab_size)
        self.cls_nsp = nn.Linear(config.hidden_size, 2)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, next_sentence_label=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        mlm_scores = self.cls_mlm(sequence_output)
        nsp_scores = self.cls_nsp(pooled_output) if next_sentence_label is not None else None
        return mlm_scores, nsp_scores

class BertPretrainingDataset(torch.utils.data.Dataset):
    """BERT预训练数据集"""
    def __init__(self, data_dir, tokenizer_name='bert-base-uncased', max_seq_length=512, num_samples=1000):
        self.tokenizer = BertTokenizer.from_pretrained("/Data/DEMO/MODEL/bert-base-uncased")
        self.max_seq_length = max_seq_length
        self.examples = self._generate_dummy_data(num_samples)
    
    def _generate_dummy_data(self, num_samples):
        """生成虚拟数据用于测试，支持MLM和NSP"""
        examples = []
        for _ in range(num_samples):
            input_ids = torch.randint(0, 30522, (self.max_seq_length,), dtype=torch.long)
            attention_mask = torch.ones(self.max_seq_length, dtype=torch.long)
            token_type_ids = torch.randint(0, 2, (self.max_seq_length,), dtype=torch.long)
            labels = torch.randint(0, 30522, (self.max_seq_length,), dtype=torch.long)
            next_sentence_label = torch.randint(0, 2, (1,), dtype=torch.long)
            
            examples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'labels': labels,
                'next_sentence_label': next_sentence_label
            })
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def train_epoch(model, dataloader, optimizer, scheduler, criterion, scaler, epoch, rank, world_size, logger, accumulation_steps):
    """训练一个epoch，支持MLM和NSP"""
    model.train()
    total_loss = 0.0
    total_steps = len(dataloader)
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)
    else:
        pbar = dataloader
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].cuda(non_blocking=True)
        attention_mask = batch['attention_mask'].cuda(non_blocking=True)
        token_type_ids = batch['token_type_ids'].cuda(non_blocking=True)
        labels = batch['labels'].cuda(non_blocking=True)
        next_sentence_label = batch['next_sentence_label'].cuda(non_blocking=True)
        
        if input_ids.dtype != torch.long or token_type_ids.dtype != torch.long or attention_mask.dtype != torch.long:
            raise ValueError(f"[Rank {rank}] Invalid input dtype: input_ids={input_ids.dtype}, token_type_ids={token_type_ids.dtype}, attention_mask={attention_mask.dtype}")
        
        with torch.cuda.amp.autocast():
            mlm_scores, nsp_scores = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, next_sentence_label=next_sentence_label)
            mlm_loss = criterion(mlm_scores.view(-1, mlm_scores.size(-1)), labels.view(-1))
            nsp_loss = criterion(nsp_scores, next_sentence_label.view(-1)) if nsp_scores is not None else torch.tensor(0.0).cuda()
            loss = (mlm_loss + nsp_loss) / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        
        total_loss += loss.item() * accumulation_steps
        
        if rank == 0 and step % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'mlm_loss': f"{mlm_loss.item():.4f}",
                'nsp_loss': f"{nsp_loss.item():.4f}" if nsp_loss != 0 else "N/A",
                'lr': f"{current_lr:.2e}"
            })
    
    avg_loss = torch.tensor(total_loss / total_steps, device='cuda')
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss.item() / world_size
    
    if rank == 0:
        logger.info(f'Epoch {epoch} - Average Loss: {avg_loss:.4f}')
    
    return avg_loss

def main():
    global args
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--local_rank', type=int, default=-1, help='本地rank')
    args = parser.parse_args()
    
    # 2. 加载配置文件（必须在使用 config 前执行）
    config = load_config(args.config)  # ✅ 优先加载 config
    
    # 3. 初始化分布式训练环境
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 4. 设置日志系统
    logger = setup_logging(rank, config['training']['log_dir'])
    
    # 5. 仅主节点创建保存目录和日志目录（避免多节点重复创建）
    if rank == 0:
        save_dir = config['training']['save_dir']
        log_dir = config['training']['log_dir']
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"已确保保存目录存在：{save_dir}")
        logger.info(f"已确保日志目录存在：{log_dir}")
        logger.info(f"开始分布式训练 - 总GPU数量: {world_size}")
        logger.info(f"本地IP: {get_local_ip()}")
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
    
    # 6. 初始化BERT模型配置
    bert_config = BertConfig(
        vocab_size=config['model']['vocab_size'],
        hidden_size=config['model']['hidden_size'],
        num_hidden_layers=config['model']['num_layers'],
        num_attention_heads=config['model']['num_heads'],
        intermediate_size=config['model']['intermediate_size'],
        max_position_embeddings=config['model']['max_seq_length'],
        hidden_dropout_prob=config['model']['dropout_rate'],
    )
    
    # 7. 初始化模型并包装DDP
    model = BertForPretraining(bert_config).cuda()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # 8. 初始化优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 9. 初始化学习率调度器
    total_steps = config['training']['epochs'] * (config['data']['steps_per_epoch'] // config['training']['accumulation_steps'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(step / (config['training']['warmup_steps'] or 1000), 1.0)
    )
    
    # 10. 初始化损失函数和混合精度
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # 11. 加载数据集
    dataset = BertPretrainingDataset(
        data_dir=config['data']['data_dir'],
        tokenizer_name=config['data']['tokenizer'],
        max_seq_length=config['model']['max_seq_length'],
        num_samples=config['data'].get('num_samples', 1000)
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    
    # 12. 训练循环
    best_loss = float('inf')
    for epoch in range(config['training']['epochs']):
        sampler.set_epoch(epoch)
        avg_loss = train_epoch(
            model, dataloader, optimizer, scheduler, criterion, scaler,
            epoch, rank, world_size, logger, config['training']['accumulation_steps']
        )
        
        # 13. 保存模型（仅主节点）
        if rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'config': config
            }
            torch.save(checkpoint, os.path.join(save_dir, 'latest_model.pth'))
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                logger.info(f"保存最佳模型 - 损失: {avg_loss:.4f}")
    
    # 14. 训练完成（仅主节点日志）
    if rank == 0:
        logger.info("训练完成！")
        logger.info(f"最佳损失: {best_loss:.4f}")
    
    # 15. 清理分布式环境
    cleanup_distributed()

if __name__ == '__main__':
    main()
