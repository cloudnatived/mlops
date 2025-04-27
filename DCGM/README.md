# DCGM、Prometheus、Grafana

https://github.com/NVIDIA/DCGM    
NVML Go bindings    https://github.com/NVIDIA/go-nvml    
DCGM Go bindings    https://github.com/NVIDIA/go-dcgm    
DCGM Exporter       https://github.com/NVIDIA/dcgm-exporter




## GPU监控工具DCGM

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
##### **1. 计算与显存指标**
| 指标               | 含义                          | 典型问题场景                         |
| ------------------ | ----------------------------- | ------------------------------------ |
| GPU Utilization    | GPU计算单元利用率（0-100%）   | 低利用率可能表示CPU或IO瓶颈。        |
| Memory Utilization | 显存使用率（0-100%）          | 显存不足会导致OOM（Out Of Memory）。 |
| FB Used/Free       | 显存已用/剩余容量（MB/GB）    | 监控模型训练时的显存占用峰值。       |
| PCIe Throughput    | PCIe接口的读写带宽（MB/s）    | 带宽不足影响数据加载速度。           |
| NVLink Throughput  | NVLink的发送/接收带宽（MB/s） | 多卡训练时通信效率低下。             |


##### **2. 硬件状态指标**
| **指标**        | **含义**                                                     | **典型问题场景**                     |
| --------------- | ------------------------------------------------------------ | ------------------------------------ |
| **Temperature** | GPU核心温度（℃）                                             | 温度过高触发降频，性能下降。         |
| **Power Usage** | 实时功耗（W）及功耗上限（TDP）                               | 功耗超标导致硬件保护性关机。         |
| **ECC Errors**  | 单比特（Correctable）和多比特（Uncorrectable）ECC内存错误计数 | ECC错误过多需更换GPU或排查环境干扰。 |
| **XID Errors**  | GPU硬件错误代码（如XID 43表示显存错误）                      | 根据XID代码定位硬件故障类型。        |


##### **3. 任务与进程级指标**
| **指标**                 | **含义**                                             | **典型问题场景**            |
| ------------------------ | ---------------------------------------------------- | --------------------------- |
| **Process Utilization**  | 各进程的GPU计算和显存使用情况                        | 识别异常占用GPU的僵尸进程。 |
| **Compute PID**          | 占用GPU计算资源的进程ID                              | 强制终止失控任务。          |
| **GPU Shared Resources** | MIG模式下各GPU实例的资源分配（如计算切片、显存切片） | 资源分配不均导致任务排队。  |



三、DCGM 的安装与使用    

```
1. 安装DCGM
# Ubuntu/Debian
# 在ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb

root@y251:/etc/apt/sources.list.d# cat cuda-ubuntu2204-x86_64.list 
deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /

apt-get update 
apt-get install -y datacenter-gpu-manager
# will install datacenter-gpu-manager 1:3.3.9

dcgmi --version
dcgmi  version: 3.3.9

默认情况下，nv-hostengine 只绑定到 127.0.0.1，因此它不会监听远程连接，也就是说无法从另一台机器获取本机信息。如果你想让它监听远程连接，需要在启动 nv-hostengine 时使用 -b 选项来指定它应该监听连接的 IP 地址。你也可以指定 -b ALL 让它监听所有网络接口上的连接。
# 停止服务
systemctl stop nvidia-dcgm
# 监听所有网络接口
nv-hostengine  --service-account nvidia-dcgm -b ALL
#获取其他节点信息
dcgmi discovery --host 10.112.0.1 -l

2. 常用命令示例
# 实时监控GPU指标（每2秒刷新）：
# -e 指定指标ID（203=GPU利用率，252=显存使用率）
# -i 指定GPU索引（0表示第一块GPU）
dcgmi dmon -i 0 -e 203,252 -c 5

# 查看GPU健康状态：
# 检查GPU 0的健康状态（-c 表示全面检测）
dcgmi health -g 0 -c

# 统计NVLink带宽：
# 显示GPU 0的NVLink状态及带宽
dcgmi nvlink -i 0 -s


2. DCGM-Exporter
DCGM-Exporter 是一种基于 NVIDIA DCGM 的 Go API 的工具，允许用户收集 GPU 指标并了解工作负载行为或监控集群中的 GPU。DCGM Exporter 是用 Go 编写的，并在 HTTP 端点 （/metrics） 上公开 GPU 指标，用于监控 Prometheus 等解决方案。
部署 DCGM Exporter：
docker run -d --gpus all --rm -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:3.3.4-3.1.5-ubuntu22.04
docker run -it -d --gpus all --name dcgm -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:3.3.0-3.2.0-ubuntu22.04 bash

进入docker：
docker start dcgm
docker exec -it dcgm  bash
-p 指定端口映射，默认端口号9400，将docker内的9400映射到主机内相同端口，即可在localhost：9400收集到数据，curl your-ip:9400/metrics 或者浏览器打开your-ip:9400/metrics有一系列指标说明成功收集到数据
-a 指定数据发送的端口

# 查看日志
cat /var/log/nv-hostengine.log
更改收集指标
https://github.com/NVIDIA/dcgm-exporter#changing-metrics
​github.com/NVIDIA/dcgm-exporter#changing-metrics
使用 dcgm-exporter，可以通过指定自定义 CSV 文件来配置要收集的字段。你可以在存储库中的 etc/default-counters.csv 下找到默认 CSV 文件，该文件将复制到您的系统或容器上的 /etc/dcgm-exporter/default-counters.csv，还可以使用 -f 选项指定自定义 csv 文件

dcgm-exporter -f /my-counters.csv

Node-Exporter
Prometheus Node Exporter 公开了各种与硬件和内核相关的指标。

Monitoring Linux host metrics with the Node Exporter | Prometheus
​prometheus.io/docs/guides/node-exporter/

https://github.com/prometheus/node_exporter/releases/download/v1.8.2/node_exporter-1.8.2.linux-amd64.tar.gz
tar xvfz node_exporter-1.8.2.linux-amd64.tar.gz
cd node_exporter-1.8.2.linux-amd64
./node_exporter

成功启动数据会暴露到9100端口
curl http://localhost:9100/metrics

3. 集成到Prometheus，Prometheus - From metrics to insight
Prometheus是一个开源的系统监控和警报工具包， 将其指标作为时间序列数据收集和存储，即指标信息与记录它的时间戳一起存储，以及称为标签的可选键值对。
下载链接：https://prometheus.io/download/
wget https://github.com/prometheus/prometheus/releases/download/v2.54.1/prometheus-2.54.1.linux-amd64.tar.gz
# 解压
tar -xzf prometheus-2.54.1.linux-amd64.tar.gz
# 打开
cd prometheus-2.54.1.linux-amd64.tar.gz

# 修改配置文件prometheus.yml(20250427)
    - job _name: 'pCGM_exporter'
      static_configs:
        - targets: ['localhost:9408','10.112.28.2:9488','10.112.57.233:9480']
    - job _name: 'node_exporter
      static_configs:
        - targets: ['localhost:9108','10.112.28.2:9100','10.112.57.233:9100']

# Prometheus配置添加Job：(20250427-ago)
    - job_name: 'dcgm'
      static_configs:
        - targets: ['gpu-node:9400']

启动服务：
./prometheus --config.file=./prometheus.yml

#查看收集结果
浏览器打开your-ip：9090，9090为prometheus的默认端口，点击status-> targets可以查看各个job的工作状态，如图所示，dcgm-exporter在三个节点均正常工作，说明收集到三个节点的信息
点击graph，勾选use local time，在搜索框内输入要查询的指标，以DCGM_FI_DEV_GPU_TEMP（GPU温度）为例，点击execute查询，table是各个指标的收集结果（文本序列），而graph可以展示一段时间内的变化情况，下图为graph的展示，15min 内的 3个节点共6张GPU的温度变化。
# 虽然prometheus提供了可视化功能，但是通常与grafana结合来建立更加全面的仪表板


选择版本及对应操作系统输入命令即可
sudo apt-get install -y adduser libfontconfig1 musl
wget https://dl.grafana.com/enterprise/release/grafana-enterprise_11.2.2_amd64.deb
dpkg -i grafana-enterprise_11.2.2_amd64.deb
确保Grafana服务已启动并且设置为开机启动，可以使用systemd来管理Grafana服务
systemctl daemon-reload

# 设置开机启动
systemctl enable grafana-server
systemctl start grafana-server

检查Grafana服务的状态：
systemctl status grafana-server

浏览器打开 your-ip：3000 进入登录界面，初始用户名与密码均为admin（grafana默认端口号3000）
导入数据源
点击 home->connections->data sources ，再选择 右上角 add new data source 添加数据源。
选择 prometheus ， 输入名字和 server URL 即可，其他根据需求设置

Grafana仪表板展示
Grafana仪表盘导入模板（官方模板ID）。
https://grafana.com/grafana/dashboards/12239-nvidia-dcgm-exporter-dashboard/

更多仪表板
node_exporter 对应的ID 为1860，效果如下：

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

数据记录与回放
记录历史指标用于事后分析：
dcgmi recorder --start -f /tmp/gpu_metrics.log
dcgmi replay -f /tmp/gpu_metrics.log

MIG监控
查看MIG实例的资源分配：
dcgmi mig -i 0 -l

六、总结
DCGM核心价值：提供从硬件状态到任务粒度的全方位GPU监控，适合数据中心级运维。
关键指标：利用率、显存、温度、NVLink/PCIe带宽、ECC/XID错误。
典型场景：性能调优、故障排查、资源调度优化。
通过DCGM，运维团队可快速定位GPU相关问题，提升集群稳定性和资源利用率。

```

