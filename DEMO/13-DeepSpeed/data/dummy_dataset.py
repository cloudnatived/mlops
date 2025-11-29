import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DummyDataset(Dataset):
    def __init__(self, vocab_size=50257, seq_length=1024, num_samples=10000):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        
        # 生成虚拟数据
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        input_ids = self.data[idx]
        
        # 创建注意力掩码（全部为1）
        attention_mask = torch.ones_like(input_ids)
        
        # 标签是输入向右移动一位
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # 忽略最后一个token的损失
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def create_data_loader(args, rank, world_size):
    """创建分布式数据加载器"""
    dataset = DummyDataset(
        vocab_size=args.vocab_size,
        seq_length=args.max_seq_length,
        num_samples=args.num_samples
    )
    
    # 分布式采样器
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader, sampler
