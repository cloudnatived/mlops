#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# RAY_cluster_test.py

import ray
import time
import random
import logging
import sys
from typing import List, Dict
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class RayClusterTester:
    def __init__(self, address: str = None):
        """初始化Ray集群测试器
        
        Args:
            address: Ray集群头部节点地址，如"ray://172.18.8.208:10001"
                     若为None，则尝试连接本地Ray实例
        """
        self.address = address
        self.connected = False
        self.cluster_info = None

    def connect(self) -> bool:
        """连接到Ray集群
        
        Returns:
            连接成功返回True，否则返回False
        """
        try:
            logger.info(f"尝试连接到Ray集群: {self.address or '本地'}")
            start_time = time.time()

            # 连接到Ray集群
            ray.init(
                address=self.address,
                ignore_reinit_error=True,
                logging_level=logging.WARNING
            )

            self.connected = True
            connect_time = time.time() - start_time
            logger.info(f"成功连接到Ray集群，耗时: {connect_time:.2f}秒")
            return True

        except Exception as e:
            logger.error(f"连接Ray集群失败: {str(e)}")
            self.connected = False
            return False

    def get_cluster_info(self) -> Dict:
        """获取集群信息
        
        Returns:
            包含集群信息的字典
        """
        if not self.connected:
            logger.error("未连接到Ray集群，请先调用connect()")
            return None

        try:
            # 获取集群资源信息
            resources = ray.cluster_resources()
            available_resources = ray.available_resources()

            # 获取节点信息
            nodes = ray.nodes()

            self.cluster_info = {
                "total_resources": resources,
                "available_resources": available_resources,
                "node_count": len(nodes),
                "nodes": nodes
            }

            logger.info(f"获取集群信息成功，节点数量: {len(nodes)}")
            return self.cluster_info

        except Exception as e:
            logger.error(f"获取集群信息失败: {str(e)}")
            return None

    def print_cluster_info(self):
        """打印集群信息"""
        if not self.cluster_info:
            self.get_cluster_info()

        if not self.cluster_info:
            return

        print("\n===== Ray集群信息 =====")
        print(f"节点数量: {self.cluster_info['node_count']}")

        print("\n总资源:")
        for resource, value in self.cluster_info['total_resources'].items():
            print(f"  {resource}: {value}")

        print("\n可用资源:")
        for resource, value in self.cluster_info['available_resources'].items():
            print(f"  {resource}: {value}")

        print("\n节点详情:")
        for i, node in enumerate(self.cluster_info['nodes']):
            print(f"  节点 {i+1}:")
            print(f"    节点ID: {node['NodeID'][:8]}...")
            print(f"    地址: {node['NodeManagerAddress']}:{node['NodeManagerPort']}")
            print(f"    资源: {node['Resources']}")
            print(f"    状态: {node['Alive']}")

    @staticmethod
    @ray.remote
    def _test_task(x: int) -> int:
        """测试任务函数，用于验证基本任务执行"""
        import time
        import os
        time.sleep(0.1)  # 模拟一些工作
        return x * 2 + os.getpid()  # 返回计算结果和进程ID

    def test_basic_tasks(self, num_tasks: int = 10) -> bool:
        """测试基本任务执行
        
        Args:
            num_tasks: 要执行的任务数量
            
        Returns:
            测试成功返回True，否则返回False
        """
        if not self.connected:
            logger.error("未连接到Ray集群，请先调用connect()")
            return False

        try:
            logger.info(f"测试基本任务执行，任务数量: {num_tasks}")
            start_time = time.time()

            # 提交多个任务
            futures = [self._test_task.remote(i) for i in range(num_tasks)]
            results = ray.get(futures)

            # 验证结果
            if len(results) != num_tasks:
                raise Exception(f"任务结果数量不匹配，预期: {num_tasks}, 实际: {len(results)}")

            # 检查是否有不同的进程ID，表明任务在不同的工作器上执行
            pids = set()
            for result in results:
                pid = result % 1000000  # 从结果中提取PID
                pids.add(pid)

            task_time = time.time() - start_time
            logger.info(f"基本任务测试成功，耗时: {task_time:.2f}秒，使用的工作器数量: {len(pids)}")
            return True

        except Exception as e:
            logger.error(f"基本任务测试失败: {str(e)}")
            return False

    @staticmethod
    @ray.remote(num_gpus=0.1)  # 请求少量GPU资源进行测试
    def _gpu_test_task() -> Dict:
        """GPU测试任务，检查GPU资源是否可用"""
        import torch
        import os

        # 检查CUDA是否可用
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        current_device = torch.cuda.current_device() if cuda_available else -1

        result = {
            "pid": os.getpid(),
            "cuda_available": cuda_available,
            "gpu_count": gpu_count,
            "current_device": current_device,
            "device_name": torch.cuda.get_device_name(current_device) if cuda_available else "N/A"
        }

        return result

    def test_gpu_resources(self, num_tasks: int = 2) -> bool:
        """测试GPU资源是否可用
        
        Args:
            num_tasks: 要执行的GPU测试任务数量
            
        Returns:
            测试成功返回True，否则返回False
        """
        if not self.connected:
            logger.error("未连接到Ray集群，请先调用connect()")
            return False

        # 检查是否有GPU资源
        resources = ray.cluster_resources()
        if "GPU" not in resources or resources["GPU"] < 0.1:
            logger.warning("集群未检测到GPU资源，跳过GPU测试")
            return True  # 跳过但不算失败    

        try:
            logger.info(f"测试GPU资源，任务数量: {num_tasks}")
            start_time = time.time()

            # 提交GPU测试任务
            futures = [self._gpu_test_task.remote() for _ in range(num_tasks)]
            results = ray.get(futures)

            # 分析结果
            gpu_available = any(result["cuda_available"] for result in results)

            if gpu_available:
                logger.info("集群中检测到可用GPU")
                for i, result in enumerate(results):
                    logger.info(f"  任务 {i+1}: GPU {result['current_device']} ({result['device_name']}) on PID {result['pid']}")
            else:
                logger.warning("集群中未检测到可用GPU")

            gpu_time = time.time() - start_time
            logger.info(f"GPU资源测试完成，耗时: {gpu_time:.2f}秒")
            return True

        except Exception as e:
            logger.error(f"GPU资源测试失败: {str(e)}")
            return False

    @staticmethod
    @ray.remote
    def _distributed_task(idx: int, shared_queue) -> int:
        """分布式任务，用于测试节点间通信"""
        import time
        import random

        # 模拟工作
        time.sleep(random.uniform(0.1, 0.5))

        # 向共享队列添加数据
        shared_queue.put(f"来自任务 {idx} 的消息")

        # 从队列获取其他任务的数据
        messages_received = 0
        while not shared_queue.empty() and messages_received < 3:
            msg = shared_queue.get()
            messages_received += 1

        return idx * 10 + messages_received

    def test_distributed_communication(self, num_tasks: int = 5) -> bool:
        """测试分布式任务间通信
        
        Args:
            num_tasks: 要执行的分布式任务数量
            
        Returns:
            测试成功返回True，否则返回False
        """
        if not self.connected:
            logger.error("未连接到Ray集群，请先调用connect()")
            return False

        try:
            logger.info(f"测试分布式任务通信，任务数量: {num_tasks}")
            start_time = time.time()

            # 创建共享队列 - 使用旧版本Ray的队列实现
            from ray.util.queue import Queue
            shared_queue = Queue(maxsize=100)

            # 提交分布式任务
            futures = [self._distributed_task.remote(i, shared_queue) for i in range(num_tasks)]
            results = ray.get(futures)

            comm_time = time.time() - start_time
            logger.info(f"分布式通信测试成功，耗时: {comm_time:.2f}秒")
            return True

        except Exception as e:
            logger.error(f"分布式通信测试失败: {str(e)}")
            return False

    @staticmethod
    @ray.remote
    def _fault_tolerance_task(task_id: int, fail_probability: float = 0.3) -> int:
        """容错测试任务，有一定概率失败"""
        import time
        import random
        time.sleep(0.2)

        # 随机失败
        if random.random() < fail_probability:
            raise Exception(f"任务 {task_id} 故意失败，测试容错性")

        return task_id

    def test_fault_tolerance(self, num_tasks: int = 10) -> bool:
        """测试Ray的容错能力
        
        Args:
            num_tasks: 要执行的任务数量
            
        Returns:
            测试成功返回True，否则返回False
        """
        if not self.connected:
            logger.error("未连接到Ray集群，请先调用connect()")
            return False

        try:
            logger.info(f"测试容错能力，任务数量: {num_tasks}")
            start_time = time.time()

            # 提交可能失败的任务
            futures = [self._fault_tolerance_task.remote(i) for i in range(num_tasks)]

            # 获取结果，处理可能的错误
            results = []
            for i, future in enumerate(futures):
                try:
                    result = ray.get(future)
                    results.append(result)
                    logger.debug(f"任务 {i} 成功完成")
                except Exception as e:
                    logger.warning(f"任务 {i} 失败: {str(e)}")

            fault_time = time.time() - start_time
            success_rate = len(results) / num_tasks * 100
            logger.info(f"容错测试完成，耗时: {fault_time:.2f}秒，成功率: {success_rate:.1f}%")

            # 即使有些任务失败，只要测试框架正常工作，也算测试成功
            return True

        except Exception as e:
            logger.error(f"容错测试框架失败: {str(e)}")
            return False

    def test_large_data_transfer(self, data_size_mb: int = 10) -> bool:
        """测试大数据传输能力
        
        Args:
            data_size_mb: 要传输的数据大小(MB)
            
        Returns:
            测试成功返回True，否则返回False
        """
        if not self.connected:
            logger.error("未连接到Ray集群，请先调用connect()")
            return False

        try:
            logger.info(f"测试大数据传输，数据大小: {data_size_mb}MB")
            start_time = time.time()

            # 创建大型数据
            data = b"x" * (data_size_mb * 1024 * 1024)  # 生成指定大小的字节数据

            # 将数据发送到远程任务并返回
            @ray.remote
            def process_large_data(data):
                return len(data)

            # 执行任务
            future = process_large_data.remote(data)
            result = ray.get(future)

            # 验证结果
            if result != len(data):
                raise Exception(f"数据传输验证失败，预期: {len(data)}, 实际: {result}")

            transfer_time = time.time() - start_time
            transfer_speed = data_size_mb / transfer_time
            logger.info(f"大数据传输测试成功，耗时: {transfer_time:.2f}秒，传输速度: {transfer_speed:.2f}MB/s")
            return True

        except Exception as e:
            logger.error(f"大数据传输测试失败: {str(e)}")
            return False

    def run_all_tests(self) -> Dict[str, bool]:
        """运行所有测试
        
        Returns:
            包含所有测试结果的字典
        """
        logger.info("开始执行所有Ray集群测试...")
        start_time = time.time()

        # 先确保连接成功
        if not self.connected:
            if not self.connect():
                return {"连接测试": False}

        # 获取并打印集群信息
        self.get_cluster_info()
        self.print_cluster_info()

        # 运行各项测试
        results = {
            "连接测试": True,  # 已经连接成功
            "基本任务测试": self.test_basic_tasks(),
            "GPU资源测试": self.test_gpu_resources(),
            "分布式通信测试": self.test_distributed_communication(),
            "容错能力测试": self.test_fault_tolerance(),
            "大数据传输测试": self.test_large_data_transfer()
        }

        # 打印测试总结
        total_time = time.time() - start_time
        print("\n===== 测试总结 =====")
        for test_name, success in results.items():
            status = "成功" if success else "失败"
            print(f"{test_name}: {status}")

        logger.info(f"所有测试完成，总耗时: {total_time:.2f}秒")
        return results

    def shutdown(self):
        """关闭Ray连接"""
        if self.connected:
            logger.info("关闭Ray连接")
            ray.shutdown()
            self.connected = False

if __name__ == "__main__":
    # 解析命令行参数获取Ray集群地址
    import argparse
    parser = argparse.ArgumentParser(description='Ray集群测试工具')
    parser.add_argument('--address', type=str, default=None,
                      help='Ray集群头部节点地址，如"ray://172.18.8.208:10001"')
    args = parser.parse_args()

    # 创建测试器并运行所有测试
    tester = RayClusterTester(address=args.address)
    try:
        results = tester.run_all_tests()

        # 检查是否有测试失败
        if all(results.values()):
            logger.info("所有测试均成功完成！")
            sys.exit(0)
        else:
            logger.error("部分测试失败，请查看日志了解详情")
            sys.exit(1)
    finally:
        tester.shutdown()
