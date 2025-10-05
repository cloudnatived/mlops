import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def demo_all_reduce(rank, world_size):
    """演示多种collective操作"""
    setup(rank, world_size)
    
    # 1. All-Reduce示例
    tensor = torch.ones(2, 3, device=f'cuda:{rank}') * (rank + 1)
    print(f"[Rank {rank}] Original tensor:\n{tensor}")
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] After all_reduce:\n{tensor}")
    
    # 2. Broadcast示例
    if rank == 0:
        broadcast_tensor = torch.tensor([1.0, 2.0, 3.0], device=f'cuda:{rank}')
    else:
        broadcast_tensor = torch.zeros(3, device=f'cuda:{rank}')
    
    dist.broadcast(broadcast_tensor, src=0)
    print(f"[Rank {rank}] After broadcast: {broadcast_tensor}")
    
    # 3. All-Gather示例
    local_tensor = torch.tensor([rank], device=f'cuda:{rank}')
    gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, local_tensor)
    print(f"[Rank {rank}] After all_gather: {[t.item() for t in gathered_tensors]}")
    
    cleanup()

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return
    
    world_size = min(4, torch.cuda.device_count())  # 限制最大GPU数
    if world_size < 1:
        print("No CUDA devices found")
        return
    
    print(f"Running distributed demo on {world_size} GPUs")
    mp.set_start_method('spawn', force=True)
    
    try:
        mp.spawn(demo_all_reduce, args=(world_size,), nprocs=world_size, join=True)
    except KeyboardInterrupt:
        print("Process interrupted")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()