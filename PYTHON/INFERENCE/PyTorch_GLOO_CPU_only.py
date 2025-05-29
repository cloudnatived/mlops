# python3 PyTorch_GLOO_CPU_only.py --backend gloo --world_size 2 --rank 0 --init_method tcp://127.0.0.1:29500
# python3 PyTorch_GLOO_CPU_only.py --backend gloo --world_size 2 --rank 1 --init_method tcp://127.0.0.1:29500


import argparse
import torch
import torch.distributed as dist

parser = argparse.ArgumentParser(description='Test CPU-only communication using GLOO backend')
parser.add_argument('--backend', type=str, default='gloo', choices=['gloo'])
parser.add_argument('--world_size', type=int, required=True)
parser.add_argument('--rank', type=int, required=True)
parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:29500')
args = parser.parse_args()

# 初始化通信组
dist.init_process_group(
    backend=args.backend,
    init_method=args.init_method,
    world_size=args.world_size,
    rank=args.rank
)

# 构造测试数据
data = torch.FloatTensor([1.0] * 128)

# 执行 all_reduce
dist.all_reduce(data, op=dist.ReduceOp.SUM)

# 检查结果
value = data.mean().item()
expected = args.world_size
assert value == expected, f"Expected {expected}, got {value}"

print("✅ PyTorch GLOO (CPU-only) communication successful!")

# 清理资源
dist.destroy_process_group()

