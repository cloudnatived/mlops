#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Test PyTorch NCCL
## 分布式测试（需要多个GPU），测试NCCL后端
#torchrun --nproc_per_node=2 test_communication.py --backend nccl

## 测试GLOO后端
#torchrun --nproc_per_node=2 test_communication.py --backend gloo

## 单GPU环境测试（跳过vLLM）
#python3 test_communication.py --backend nccl --skip-vllm

## 分布式测试，跳过vLLM
#torchrun --nproc_per_node=2 test_communication.py --backend nccl --skip-vllm

# 3机互连通信检查，每个node节点运行一样的命令
# NCCL_DEBUG=TRACE torchrun --nnodes 3 --nproc-per-node=8 --rdzv_backend=c10d --rdzv_endpoint=head_node_ip:8887 check_nccl.py

import argparse
import os
import torch
import torch.distributed as dist

def setup_distributed():
    """设置分布式环境"""
    parser = argparse.ArgumentParser(description='Test communication backends')
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo', 'mpi'])
    parser.add_argument('--skip-vllm', action='store_true', help='Skip vLLM NCCL tests')
    args = parser.parse_args()
    
    # 多节点环境使用环境变量初始化
    if not dist.is_initialized():
        dist.init_process_group(backend=args.backend)
    
    return args

def test_basic_communication():
    """测试基础通信"""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    print(f"Rank {rank}/{world_size}, Local Rank {local_rank}: Starting communication test")
    
    # 设置设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    # 测试数据
    data = torch.ones(128, device=device)
    
    # 同步所有进程
    dist.barrier()
    
    # All-reduce 测试
    dist.all_reduce(data, op=dist.ReduceOp.SUM)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    value = data.mean().item()
    expected = world_size
    
    if rank == 0:
        print(f"All-reduce result: {value}, Expected: {expected}")
    
    assert abs(value - expected) < 1e-6, f"Expected {expected}, got {value}"
    
    return world_size, rank, local_rank

def test_gloo_backend(world_size):
    """测试GLOO后端"""
    if world_size <= 1:
        return
    
    # 创建GLOO组
    gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    
    # 在CPU上测试GLOO
    cpu_data = torch.ones(128)
    dist.all_reduce(cpu_data, op=dist.ReduceOp.SUM, group=gloo_group)
    value = cpu_data.mean().item()
    
    assert abs(value - world_size) < 1e-6, f"GLOO: Expected {world_size}, got {value}"
    
    if dist.get_rank() == 0:
        print("PyTorch GLOO is successful!")
    
    dist.destroy_process_group(gloo_group)

def test_vllm_nccl(world_size, local_rank, skip_vllm=False):
    """测试vLLM NCCL（可选）"""
    if skip_vllm or world_size <= 1:
        if dist.get_rank() == 0:
            print("Skipping vLLM tests")
        return
    
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        
        # 创建GLOO组用于vLLM
        gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        
        pynccl = PyNcclCommunicator(group=gloo_group, device=local_rank)
        pynccl.disabled = False
        
        # 测试vLLM NCCL
        data = torch.ones(128, device=f"cuda:{local_rank}")
        
        # 使用CUDA stream
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            data.fill_(1)
            out = pynccl.all_reduce(data, stream=s)
            value = out.mean().item()
        
        assert abs(value - world_size) < 1e-6, f"vLLM NCCL: Expected {world_size}, got {value}"
        
        if dist.get_rank() == 0:
            print("vLLM NCCL is successful!")
        
        # 测试CUDA Graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph=g, stream=s):
            out = pynccl.all_reduce(data, stream=torch.cuda.current_stream())
        
        data.fill_(1)
        g.replay()
        torch.cuda.current_stream().synchronize()
        value = out.mean().item()
        
        assert abs(value - world_size) < 1e-6, f"vLLM CUDA Graph: Expected {world_size}, got {value}"
        
        if dist.get_rank() == 0:
            print("vLLM NCCL with CUDA Graph is successful!")
        
        dist.destroy_process_group(gloo_group)
        
    except ImportError:
        if dist.get_rank() == 0:
            print("vLLM not available, skipping vLLM tests")

def main():
    """主函数"""
    args = setup_distributed()
    
    try:
        world_size, rank, local_rank = test_basic_communication()
        
        if rank == 0:
            print("PyTorch NCCL is successful!")
        
        test_gloo_backend(world_size)
        test_vllm_nccl(world_size, local_rank, args.skip_vllm)
        
        if rank == 0:
            print("All tests completed successfully!")
            
    except Exception as e:
        print(f"Rank {dist.get_rank()}: Error - {e}")
        raise
    
    finally:
        # 清理
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
