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
import torch
import torch.distributed as dist

parser = argparse.ArgumentParser(description='Test communication backends')
parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo', 'mpi'])
parser.add_argument('--skip-vllm', action='store_true', help='Skip vLLM NCCL tests')
args = parser.parse_args()

# 单机测试时初始化进程组
if not dist.is_initialized():
    dist.init_process_group(backend=args.backend, init_method='tcp://localhost:23456', rank=0, world_size=1)

local_rank = dist.get_rank() % torch.cuda.device_count()
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

# 测试基础通信
if torch.cuda.is_available():
    data = torch.FloatTensor([1,] * 128).to("cuda")
    dist.all_reduce(data, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    value = data.mean().item()
else:
    data = torch.FloatTensor([1,] * 128)
    dist.all_reduce(data, op=dist.ReduceOp.SUM)
    value = data.mean().item()

world_size = dist.get_world_size()
assert value == world_size, f"Expected {world_size}, got {value}"

print("PyTorch NCCL is successful!")

# Test PyTorch GLOO
gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
cpu_data = torch.FloatTensor([1,] * 128)
dist.all_reduce(cpu_data, op=dist.ReduceOp.SUM, group=gloo_group)
value = cpu_data.mean().item()
assert value == world_size, f"Expected {world_size}, got {value}"

print("PyTorch GLOO is successful!")

# 如果跳过vLLM测试或者world_size<=1，则退出
if args.skip_vllm or world_size <= 1:
    print("Skipping vLLM tests as requested")
    dist.destroy_process_group(gloo_group)
    dist.destroy_process_group()
    exit()

# Test vLLM NCCL, with cuda graph
try:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    pynccl = PyNcclCommunicator(group=gloo_group, device=local_rank)
    # pynccl is enabled by default for 0.6.5+,
    # but for 0.6.4 and below, we need to enable it manually.
    # keep the code for backward compatibility when because people
    # prefer to read the latest documentation.
    pynccl.disabled = False

    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        data.fill_(1)
        out = pynccl.all_reduce(data, stream=s)
        value = out.mean().item()
        assert value == world_size, f"Expected {world_size}, got {value}"

    print("vLLM NCCL is successful!")

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph=g, stream=s):
        out = pynccl.all_reduce(data, stream=torch.cuda.current_stream())

    data.fill_(1)
    g.replay()
    torch.cuda.current_stream().synchronize()
    value = out.mean().item()
    assert value == world_size, f"Expected {world_size}, got {value}"

    print("vLLM NCCL with cuda graph is successful!")

except ImportError:
    print("vLLM not available, skipping vLLM tests")

dist.destroy_process_group(gloo_group)
dist.destroy_process_group()
