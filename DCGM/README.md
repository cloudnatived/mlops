### DCGM、Prometheus、Grafana

https://github.com/NVIDIA/DCGM    
NVML Go bindings    https://github.com/NVIDIA/go-nvml    
DCGM Go bindings    https://github.com/NVIDIA/go-dcgm    
DCGM Exporter       https://github.com/NVIDIA/dcgm-exporter




## GPU监控工具DCGM
```
NVIDIA DCGM（Data Center GPU Manager） 是专为数据中心设计的GPU监控与管理工具，支持实时监控、性能分析和自动化运维。以下是DCGM的核心功能及其监控的GPU关键指标详解：

一、DCGM 的核心功能

    集群级GPU监控
        支持多节点、多GPU的统一监控，适用于大规模AI训练、HPC等场景。
        提供API和命令行工具（dmon、dstat），可集成到Prometheus、Grafana等平台。

    健康状态检测
        实时检测GPU硬件健康状态（如XID错误、ECC内存错误）。
        自动触发告警或恢复操作（如重启服务、隔离故障GPU）。

    性能分析与优化
        统计GPU利用率、显存使用率等指标，定位性能瓶颈。
        支持NVLink/PCIe带宽分析，优化多卡通信。

    策略管理
        设置功耗/温度阈值，防止GPU过热或过载。
        支持MIG（Multi-Instance GPU）资源的细粒度监控。

二、DCGM 监控的核心GPU指标

以下指标可通过 dcgmi dmon 或 nvidia-smi 查看，并支持通过API获取：
1. 计算与显存指标


##### **1. 计算与显存指标**

| 指标               | 含义                          | 典型问题场景                         |
| ------------------ | ----------------------------- | ------------------------------------ |
| GPU Utilization    | GPU计算单元利用率（0-100%）   | 低利用率可能表示CPU或IO瓶颈。        |
| Memory Utilization | 显存使用率（0-100%）          | 显存不足会导致OOM（Out Of Memory）。 |
| FB Used/Free       | 显存已用/剩余容量（MB/GB）    | 监控模型训练时的显存占用峰值。       |
| PCIe Throughput    | PCIe接口的读写带宽（MB/s）    | 带宽不足影响数据加载速度。           |
| NVLink Throughput  | NVLink的发送/接收带宽（MB/s） | 多卡训练时通信效率低下。             |






指标	含义	典型问题场景
GPU Utilization	GPU计算单元利用率（0-100%）	低利用率可能表示CPU或IO瓶颈。
Memory Utilization	显存使用率（0-100%）	显存不足会导致OOM（Out Of Memory）。
FB Used/Free	显存已用/剩余容量（MB/GB）	监控模型训练时的显存占用峰值。
PCIe Throughput	PCIe接口的读写带宽（MB/s）	带宽不足影响数据加载速度。
NVLink Throughput	NVLink的发送/接收带宽（MB/s）	多卡训练时通信效率低下。


2. 硬件状态指标
指标	含义	典型问题场景
Temperature	GPU核心温度（℃）	温度过高触发降频，性能下降。
Power Usage	实时功耗（W）及功耗上限（TDP）	功耗超标导致硬件保护性关机。
ECC Errors	单比特（Correctable）和多比特（Uncorrectable）ECC内存错误计数	ECC错误过多需更换GPU或排查环境干扰。
XID Errors	GPU硬件错误代码（如XID 43表示显存错误）	根据XID代码定位硬件故障类型。
3. 任务与进程级指标
指标	含义	典型问题场景
Process Utilization	各进程的GPU计算和显存使用情况	识别异常占用GPU的僵尸进程。
Compute PID	占用GPU计算资源的进程ID	强制终止失控任务。
GPU Shared Resources	MIG模式下各GPU实例的资源分配（如计算切片、显存切片）	资源分配不均导致任务排队。
三、DCGM 的安装与使用
1. 安装DCGM

# Ubuntu/Debian
apt-get install -y datacenter-gpu-manager
dcgmi --version

# RHEL/CentOS
yum install -y datacenter-gpu-manager
systemctl enable nvidia-dcgm
systemctl start nvidia-dcgm

    1
    2
    3
    4
    5
    6
    7
    8

2. 常用命令示例

    实时监控GPU指标（每2秒刷新）：

    dcgmi dmon -i 0 -e 203,252 -c 5
    # -e 指定指标ID（203=GPU利用率，252=显存使用率）
    # -i 指定GPU索引（0表示第一块GPU）
        1
        2
        3

    查看GPU健康状态：

    dcgmi health -g 0 -c
    # 检查GPU 0的健康状态（-c 表示全面检测）
        1
        2

    统计NVLink带宽：

    dcgmi nvlink -i 0 -s
    # 显示GPU 0的NVLink状态及带宽
        1
        2

3. 集成到Prometheus

    部署 DCGM Exporter：

    docker run -d --gpus all --rm -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:3.3.4-3.1.5-ubuntu22.04
        1

    Prometheus配置添加Job：

    - job_name: 'dcgm'
      static_configs:
        - targets: ['gpu-node:9400']
        1
        2
        3

    Grafana仪表盘导入模板（官方模板ID）。

四、典型问题排查案例
案例1：GPU利用率低

    现象：训练任务中GPU利用率长期低于30%。
    排查步骤：
        dcgmi dmon 观察 GPU Utilization 和 PCIe Throughput。
        若PCIe读取带宽高但GPU利用率低，可能是数据预处理（CPU）成为瓶颈。
        优化方法：增加数据加载线程、启用DALI加速或使用更快的存储。

案例2：显存不足（OOM）

    现象：任务运行时出现CUDA OOM错误。
    排查步骤：
        dcgmi dmon -e 252 监控显存使用峰值。
        调整Batch Size或使用梯度累积。
        启用混合精度训练（AMP）减少显存占用。

案例3：NVLink带宽低

    现象：多卡训练速度未随GPU数量线性提升。
    排查步骤：
        dcgmi nvlink -s 检查NVLink带宽是否达到预期（如A100 NVLink3为600GB/s）。
        确认拓扑是否对称（nvidia-smi topo -m）。
        设置NCCL参数：export NCCL_ALGO=Tree。

五、DCGM 的高级功能

    策略引擎（Policy Engine）
        自动响应GPU事件（如温度超限时降低功耗）。

    dcgmi policy --group my_policy --set "temperature,action=throttle,threshold=90"
        1

    数据记录与回放
        记录历史指标用于事后分析：

        dcgmi recorder --start -f /tmp/gpu_metrics.log
        dcgmi replay -f /tmp/gpu_metrics.log
            1
            2

    MIG监控
        查看MIG实例的资源分配：

        dcgmi mig -i 0 -l
            1

六、总结

    DCGM核心价值：提供从硬件状态到任务粒度的全方位GPU监控，适合数据中心级运维。
    关键指标：利用率、显存、温度、NVLink/PCIe带宽、ECC/XID错误。
    典型场景：性能调优、故障排查、资源调度优化。
    通过DCGM，运维团队可快速定位GPU相关问题，提升集群稳定性和资源利用率。

参考资料：
GPU监控工具DCGM    https://blog.csdn.net/Franklin7B/article/details/145585589

```
