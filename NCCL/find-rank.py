import os
import torch
import torch.distributed as dist
import nccl

# 假设已经设置了环境变量RANK来表示rank
rank = int(os.environ.get("RANK"))
world_size = 4  # 假设总共有4个进程（GPU）
# 初始化分布式环境（这里只是简单示意，实际还需要更多设置）
dist.init_process_group("nccl", rank = rank, world_size = world_size)
# 初始化NCCL通信（假设已经有合适的模型和数据）
nccl_comm = nccl.Communicator()
nccl_comm.init_rank(world_size, rank)
# 后续进行NCCL通信操作和训练
#...
# 清理NCCL通信和分布式环境
nccl_comm.destroy()
dist.destroy_process_group()
