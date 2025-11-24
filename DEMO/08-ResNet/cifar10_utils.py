#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# cifar10_utils.py
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import os

# CIFAR-10 官方数据集 URL 和文件名（用于检查）
_CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_CIFAR10_FILENAME = "cifar-10-python.tar.gz"
_CIFAR10_EXTRACTED_DIR = "cifar-10-batches-py"


def setup_distributed(rank, world_size, master_addr, port="29500"):
    """初始化分布式训练"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())


def cleanup():
    dist.destroy_process_group()


def _check_cifar10_manual_download(data_dir):
    """
    检查 data_dir 目录下是否存在 cifar-10-python.tar.gz 文件
    如果存在，返回 True（表示不需要下载）
    """
    tar_path = os.path.join(data_dir, _CIFAR10_FILENAME)
    extracted_path = os.path.join(data_dir, _CIFAR10_EXTRACTED_DIR)
    # 如果 .tar.gz 存在 或 已解压的目录存在，都认为数据已准备
    return os.path.isfile(tar_path) or os.path.isdir(extracted_path)


def get_dataloaders(batch_size, data_dir='./data'):
    """
    获取 CIFAR-10 数据集
    :param batch_size: 每卡 batch size
    :param data_dir: 数据集保存路径（相对或绝对路径）
    """
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # ======== 分布式安全：检查数据是否存在 ========
    is_main_process = (dist.get_rank() == 0)
    world_size = dist.get_world_size()

    # 主进程检查是否需要下载
    should_download = False
    if is_main_process:
        if not _check_cifar10_manual_download(data_dir):
            should_download = True
            print(f"🔍 Rank 0: {_CIFAR10_FILENAME} not found. Will download from {_CIFAR10_URL}")
        else:
            print(f"✅ Rank 0: {_CIFAR10_FILENAME} or extracted data found. Skip download.")
    
    # 广播 should_download 标志给所有进程
    should_download_tensor = torch.tensor(int(should_download), dtype=torch.uint8, device='cuda')
    dist.broadcast(should_download_tensor, src=0)
    should_download = bool(should_download_tensor.item())

    # 所有进程等待主进程完成检查
    dist.barrier()

    # ======== 加载数据集 ========
    # 无论是否下载，所有进程都用 download=should_download
    # PyTorch 内部会处理：如果已解压，即使 download=True 也不会重复下载
    train_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=should_download,
        transform=transform_train
    )
    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=should_download,
        transform=transform_test
    )

    # 再次同步，确保所有进程完成数据加载
    dist.barrier()

    # 分布式采样器
    train_sampler = DistributedSampler(train_set, shuffle=True)
    test_sampler = DistributedSampler(test_set, shuffle=False)

    # DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, test_loader, train_sampler
