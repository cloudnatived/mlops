# NCCL_multi_node_nccl_test.py

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta

def setup_distributed(rank, world_size, master_addr, master_port):
    """设置分布式环境"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['NCCL_DEBUG'] = 'INFO'  # 调试信息
    #os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # 根据实际网卡调整
    os.environ['NCCL_SOCKET_IFNAME'] = 'enp4s1'  # 根据实际网卡调整
    
    # 初始化进程组
    dist.init_process_group(
        "nccl", 
        rank=rank, 
        world_size=world_size,
        timeout=timedelta(seconds=60)
    )
    
    # 设置当前GPU设备
    torch.cuda.set_device(rank % torch.cuda.device_count())
    print(f"[Rank {rank}] Initialized on node {os.uname()[1]}")

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def run_benchmark(rank, world_size, master_addr, master_port, iterations=10):
    """运行NCCL性能测试"""
    try:
        setup_distributed(rank, world_size, master_addr, master_port)
        
        # 获取本地和全局信息
        local_rank = rank % torch.cuda.device_count()
        node_name = os.uname()[1]
        
        print(f"[Rank {rank}] Node: {node_name}, Local GPU: {local_rank}, World Size: {world_size}")
        
        # 测试不同大小的张量
        sizes = [2**i for i in range(10, 25)]  # 1KB 到 32MB
        
        results = []
        for size in sizes:
            try:
                # 创建测试张量
                tensor = torch.randn(size // 4, device=f'cuda:{local_rank}')  # float32占4字节
                
                # 预热
                for _ in range(3):
                    dist.all_reduce(tensor.clone(), op=dist.ReduceOp.SUM)
                
                # 计时测试
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                for _ in range(iterations):
                    dist.all_reduce(tensor.clone(), op=dist.ReduceOp.SUM)
                end_event.record()
                
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event) / iterations
                
                bandwidth = (size * world_size * 4) / (elapsed_time * 1e6)  # GB/s
                
                results.append((size, elapsed_time, bandwidth))
                print(f"[Rank {rank}] Size: {size:>10} bytes, Time: {elapsed_time:.3f} ms, Bandwidth: {bandwidth:.2f} GB/s")
                
            except Exception as e:
                print(f"[Rank {rank}] Error testing size {size}: {e}")
                continue
        
        # 收集所有节点的结果
        if rank == 0:
            print("\n" + "="*80)
            print("NCCL PERFORMANCE SUMMARY")
            print("="*80)
            print(f"{'Size (bytes)':<15} {'Time (ms)':<12} {'Bandwidth (GB/s)':<20} {'World Size':<12}")
            print("-"*80)
            for size, time_ms, bandwidth in results:
                print(f"{size:<15} {time_ms:<12.3f} {bandwidth:<20.2f} {world_size:<12}")
        
        return results
        
    except Exception as e:
        print(f"[Rank {rank}] Error in benchmark: {e}")
        return []
    finally:
        cleanup()

def main():
    if len(sys.argv) != 5:
        print("Usage: python multi_node_nccl_test.py <rank> <world_size> <master_addr> <master_port>")
        print("Example: python multi_node_nccl_test.py 0 4 192.168.1.100 29500")
        sys.exit(1)
    
    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])
    master_addr = sys.argv[3]
    master_port = int(sys.argv[4])
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print(f"[Rank {rank}] CUDA is not available")
        return
    
    print(f"[Rank {rank}] Starting NCCL test...")
    print(f"[Rank {rank}] Master: {master_addr}:{master_port}")
    
    try:
        results = run_benchmark(rank, world_size, master_addr, master_port)
        print(f"[Rank {rank}] Test completed successfully")
    except Exception as e:
        print(f"[Rank {rank}] Test failed: {e}")
        raise

if __name__ == "__main__":
    main()
