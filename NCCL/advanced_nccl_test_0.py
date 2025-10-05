import os
import sys
import time
import torch
import torch.distributed as dist
import numpy as np
from datetime import timedelta

class NCCLEvaluator:
    def __init__(self, rank, world_size, master_addr, master_port):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.local_rank = rank % torch.cuda.device_count()
        
    def setup(self):
        """设置分布式环境"""
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = str(self.master_port)
        os.environ['NCCL_TIMEOUT'] = '1800'  # 30分钟超时
        os.environ['NCCL_DEBUG'] = 'WARN'  # 减少调试输出
        
        dist.init_process_group(
            "nccl",
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=1800)
        )
        
        torch.cuda.set_device(self.local_rank)
        print(f"[Rank {self.rank}] Setup completed on {os.uname()[1]}")

    def cleanup(self):
        """清理环境"""
        dist.destroy_process_group()

    def test_collective_operations(self):
        """测试各种集合通信操作"""
        operations = {
            'all_reduce': self.test_all_reduce,
            'all_gather': self.test_all_gather,
            'reduce_scatter': self.test_reduce_scatter,
            'broadcast': self.test_broadcast,
            'all_to_all': self.test_all_to_all
        }
        
        results = {}
        for op_name, op_func in operations.items():
            try:
                print(f"[Rank {self.rank}] Testing {op_name}...")
                result = op_func()
                results[op_name] = result
                print(f"[Rank {self.rank}] {op_name} completed: {result}")
            except Exception as e:
                print(f"[Rank {self.rank}] {op_name} failed: {e}")
                results[op_name] = None
                
        return results

    def test_all_reduce(self, size=1024*1024):
        """测试All-Reduce操作"""
        tensor = torch.randn(size // 4, device=f'cuda:{self.local_rank}')
        
        # 预热
        for _ in range(5):
            dist.all_reduce(tensor.clone(), op=dist.ReduceOp.SUM)
        
        # 测试
        times = []
        for _ in range(10):
            start = time.time()
            dist.all_reduce(tensor.clone(), op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        bandwidth = (size * self.world_size * 4) / (avg_time * 1e9)  # GB/s
        return {'time': avg_time, 'bandwidth': bandwidth}

    def test_all_gather(self, size=1024*1024):
        """测试All-Gather操作"""
        local_tensor = torch.randn(size // (4 * self.world_size), device=f'cuda:{self.local_rank}')
        gathered_tensors = [torch.empty_like(local_tensor) for _ in range(self.world_size)]
        
        # 预热
        for _ in range(5):
            dist.all_gather(gathered_tensors, local_tensor.clone())
        
        # 测试
        times = []
        for _ in range(10):
            start = time.time()
            dist.all_gather(gathered_tensors, local_tensor.clone())
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        bandwidth = (size * self.world_size) / (avg_time * 1e9)
        return {'time': avg_time, 'bandwidth': bandwidth}

    def test_reduce_scatter(self, size=1024*1024):
        """测试Reduce-Scatter操作"""
        input_tensor = torch.randn(size // 4, device=f'cuda:{self.local_rank}')
        output_tensor = torch.empty(size // (4 * self.world_size), device=f'cuda:{self.local_rank}')
        
        # 预热
        for _ in range(5):
            dist.reduce_scatter(output_tensor.clone(), list(input_tensor.chunk(self.world_size)))
        
        # 测试
        times = []
        for _ in range(10):
            start = time.time()
            dist.reduce_scatter(output_tensor.clone(), list(input_tensor.chunk(self.world_size)))
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        bandwidth = (size * self.world_size) / (avg_time * 1e9)
        return {'time': avg_time, 'bandwidth': bandwidth}

    def test_broadcast(self, size=1024*1024):
        """测试Broadcast操作"""
        tensor = torch.randn(size // 4, device=f'cuda:{self.local_rank}')
        
        # 预热
        for _ in range(5):
            dist.broadcast(tensor.clone(), src=0)
        
        # 测试
        times = []
        for _ in range(10):
            start = time.time()
            dist.broadcast(tensor.clone(), src=0)
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        bandwidth = (size * self.world_size) / (avg_time * 1e9)
        return {'time': avg_time, 'bandwidth': bandwidth}

    def test_all_to_all(self, size=1024*1024):
        """测试All-to-All操作"""
        try:
            chunk_size = size // (4 * self.world_size)
            input_list = [torch.randn(chunk_size, device=f'cuda:{self.local_rank}') for _ in range(self.world_size)]
            output_list = [torch.empty(chunk_size, device=f'cuda:{self.local_rank}') for _ in range(self.world_size)]
            
            # 预热
            for _ in range(5):
                dist.all_to_all(output_list, input_list)
            
            # 测试
            times = []
            for _ in range(10):
                start = time.time()
                dist.all_to_all(output_list, input_list)
                torch.cuda.synchronize()
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            bandwidth = (size * self.world_size) / (avg_time * 1e9)
            return {'time': avg_time, 'bandwidth': bandwidth}
        except Exception as e:
            return {'error': str(e)}

    def run_comprehensive_test(self):
        """运行综合测试"""
        try:
            self.setup()
            
            print(f"[Rank {self.rank}] Starting comprehensive NCCL test...")
            
            # 基础连通性测试
            test_tensor = torch.tensor([self.rank], device=f'cuda:{self.local_rank}', dtype=torch.float32)
            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            expected_sum = sum(range(self.world_size))
            
            if abs(test_tensor.item() - expected_sum) < 1e-6:
                print(f"[Rank {self.rank}] Basic connectivity test passed!")
            else:
                print(f"[Rank {self.rank}] Basic connectivity test failed!")
                return False
            
            # 性能测试
            results = self.test_collective_operations()
            
            # 汇总结果
            if self.rank == 0:
                self.print_summary(results)
            
            return True
            
        except Exception as e:
            print(f"[Rank {self.rank}] Test failed with error: {e}")
            return False
        finally:
            self.cleanup()

    def print_summary(self, results):
        """打印测试摘要"""
        print("\n" + "="*80)
        print("NCCL COMPREHENSIVE TEST RESULTS")
        print("="*80)
        print(f"Nodes: {self.world_size // torch.cuda.device_count_per_node() if hasattr(torch.cuda, 'device_count_per_node') else 'Unknown'}")
        print(f"Total GPUs: {self.world_size}")
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*80)
        
        for op_name, result in results.items():
            if result and 'error' not in result:
                print(f"{op_name.upper():<15} Time: {result['time']*1000:.3f}ms Bandwidth: {result['bandwidth']:.2f} GB/s")
            elif result and 'error' in result:
                print(f"{op_name.upper():<15} Error: {result['error']}")
            else:
                print(f"{op_name.upper():<15} Failed")

def main():
    if len(sys.argv) != 5:
        print("Usage: python advanced_nccl_test.py <rank> <world_size> <master_addr> <master_port>")
        sys.exit(1)
    
    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])
    master_addr = sys.argv[3]
    master_port = int(sys.argv[4])
    
    if not torch.cuda.is_available():
        print(f"[Rank {rank}] CUDA not available")
        sys.exit(1)
    
    evaluator = NCCLEvaluator(rank, world_size, master_addr, master_port)
    
    try:
        success = evaluator.run_comprehensive_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"[Rank {rank}] Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"[Rank {rank}] Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()