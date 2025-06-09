#
## 测试NCCL后端
#torchrun --nproc_per_node=2 test_communication.py --backend nccl

## 测试GLOO后端
#torchrun --nproc_per_node=2 test_communication.py --backend gloo

## 单GPU环境测试（跳过vLLM）
#python test_communication.py --backend nccl --skip-vllm

# 3机互连通信检查，每个node节点运行一样的命令
# NCCL_DEBUG=TRACE torchrun --nnodes 3 --nproc-per-node=8 --rdzv_backend=c10d --rdzv_endpoint=head_node_ip:8887 check_nccl.py

# Test PyTorch NCCL

import torch
import torch.distributed as dist

parser = argparse.ArgumentParser(description='Test communication backends')
parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo', 'mpi'])
args = parser.parse_args()

# 初始化时使用命令行指定的后端
dist.init_process_group(backend=args.backend)
#dist.init_process_group(backend="nccl")

local_rank = dist.get_rank() % torch.cuda.device_count()
torch.cuda.set_device(local_rank)
data = torch.FloatTensor([1,] * 128).to("cuda")
dist.all_reduce(data, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
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

if world_size <= 1:
    exit()

# Test vLLM NCCL, with cuda graph
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

dist.destroy_process_group(gloo_group)
dist.destroy_process_group()
