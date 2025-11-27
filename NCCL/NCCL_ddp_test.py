# NCCL_ddp_test.py
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    t = torch.ones(1).cuda(rank) * (rank + 1)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] result = {t.item()}")

    dist.destroy_process_group()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
