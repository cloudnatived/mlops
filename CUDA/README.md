# 一. 英伟达GPU驱动、CUDA Toolkit、Nccl

```
https://developer.nvidia.com/nccl/nccl-download

Download NCCL 2.26.2, for CUDA 12.8, March 13th, 2025
Download NCCL 2.26.2, for CUDA 12.4, March 13th, 2025   #选择所需版本
Download NCCL 2.26.2, for CUDA 12.2, March 13th, 2025

# 选择下载安装
https://developer.nvidia.com/downloads/compute/machine-learning/nccl/secure/2.26.2/ubuntu2204/x86_64/nccl-local-repo-ubuntu2204-2.26.2-cuda12.4_1.0-1_amd64.deb

# 安装
root@y248:/Data# dpkg -i nccl-local-repo-ubuntu2204-2.26.2-cuda12.4_1.0-1_amd64.deb 
Selecting previously unselected package nccl-local-repo-ubuntu2204-2.26.2-cuda12.4.
(Reading database ... 171185 files and directories currently installed.)
Preparing to unpack nccl-local-repo-ubuntu2204-2.26.2-cuda12.4_1.0-1_amd64.deb ...
Unpacking nccl-local-repo-ubuntu2204-2.26.2-cuda12.4 (1.0-1) ...
Setting up nccl-local-repo-ubuntu2204-2.26.2-cuda12.4 (1.0-1) ...

The public nccl-local-repo-ubuntu2204-2.26.2-cuda12.4 GPG key does not appear to be installed.
To install the key, run this command:
sudo cp /var/nccl-local-repo-ubuntu2204-2.26.2-cuda12.4/nccl-local-960AB412-keyring.gpg /usr/share/keyrings/

# 查看包安装情况
dpkg -l|grep nccl

# dpkg 查看软件包的安装位置
dpkg -L nccl-local-repo-ubuntu2204-2.26.2-cuda12.4



# 选择网络安装
Network Installer for Ubuntu22.04
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
$ sudo dpkg -i cuda-keyring_1.1-1_all.deb
$ sudo apt-get update



make CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local/lib/python3.10/dist-packages/nvidia/nccl -lnccl
```

# 二. 容器化工具nvidia-container-toolkit

```
NVIDIA Container Toolkit 使用户能够构建和运行 GPU 加速容器。该工具包包括一个容器运行时库和实用程序，用于自动配置容器以利用 NVIDIA GPU。
安装nvidia-container-toolkitde 的前提先安装好docker和nvidia驱动。

1. 配置存储库
英伟达官方存储库配置：
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
sed ‘s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g’ |
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

由于官网的放在github上，访问很慢所以这里使用国内的存储库，中科大的源：
curl -fsSL https://mirrors.ustc.edu.cn/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://mirrors.ustc.edu.cn/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://nvidia.github.io#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://mirrors.ustc.edu.cn#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

2. 更新软件包列表
apt-get update

3. 安装nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit

4. 验证安装
nvidia-container-cli  --version

三、 配置
1. 配置docker
使用 nvidia-ctk 命令配置容器运行时：
该命令用于配置 Docker 以使用 NVIDIA 容器运行时。具体来说，它会修改 /etc/docker/daemon.json 文件，将 NVIDIA 容器运行时设置为 Docker 的默认运行时
配置 Docker 使用 NVIDIA 容器运行时：这允许 Docker 容器访问和利用 NVIDIA GPU 资源，从而支持 GPU 加速。
修改 /etc/docker/daemon.json 文件：该命令会将 NVIDIA 容器运行时的配置信息写入 Docker 的配置文件中。

$ sudo nvidia-ctk runtime configure --runtime=docker
INFO[0000] Loading config from /etc/docker/daemon.json
INFO[0000] Wrote updated config to /etc/docker/daemon.json
INFO[0000] It is recommended that docker daemon be restarted.

重启docker
systemctl restart docker

$ cat /etc/docker/daemon.json
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}

查看docker 支持的运行时有没有nvidia
$ docker info | grep Runtimes
 Runtimes: nvidia runc io.containerd.runc.v2

四、 启动容器运行 nvidia-smi 查看效果
$ sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
--runtime=nvidia : 指定容器运行时
--gpus all：请求所有可用的 GPU 资源
nvidia-smi：查看 NVIDIA GPU 的状态信息，包括 GPU 使用率、内存使用情况等

$ sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

# 三. k8s-device-plugin 和 node-feature-discovery

kubernetes集群管理GPU节点和作业的2个主要组件，k8s-device-plugin和node-feature-discovery

![](IMAGE-1/MLOps-kubernetes+GPU.png)

#### 2.4.1.nvidia/k8s-device-plugin

Github：https://github.com/NVIDIA/k8s-device-plugin

**Device Plugin 机制**：从 Kubernetes 1.8 开始，官方提供了可插拔的 Device Plugin 框架。厂商可以自己实现一个 Device Plugin —— 一个自定义的守护进程（daemon），通过 gRPC 与 kubelet 通信，把硬件信息“上报”给 kubelet。kubelet 不必自带厂商逻辑，只管和这个 Device Plugin 标准对接即可。深入浅出 K8s 设备插件技术（Device Plugin）。 nvidia/k8s-device-plugin以 DaemonSet 的方式部署到集群每个节点上，让节点上的 GPU 被 kubelet 发现、上报给 Kubernetes。然后，请求在 Pod spec 里声明 `resources.limits.nvidia.com/gpu: 1` ，就可以使用对应节点上的 GPU 了。



#### 2.4.2.Node Feature Discovery（node-feature-discovery）

Github：https://github.com/kubernetes-sigs/node-feature-discovery
Docs：https://kubernetes-sigs.github.io/node-feature-discovery/master/get-started/index.html

Node Feature Discovery（NFD）是由Intel创建的项目，能够帮助Kubernetes集群更智能地管理节点资源。它通过检测每个节点的特性能力（例如CPU型号、GPU型号、内存大小等）并将这些能力以标签的形式发送到Kubernetes集群的API服务器（kube-apiserver）。然后，通过kube-apiserver修改节点的标签。这些标签可以帮助调度器（kube-scheduler）更智能地选择最适合特定工作负载的节点来运行Pod。

```
$ kubectl apply -k "https://github.com/kubernetes-sigs/node-feature-discovery/deployment/overlays/default?ref=v0.17.2"
  namespace/node-feature-discovery created
  customresourcedefinition.apiextensions.k8s.io/nodefeaturegroups.nfd.k8s-sigs.io created
  customresourcedefinition.apiextensions.k8s.io/nodefeaturerules.nfd.k8s-sigs.io created
  customresourcedefinition.apiextensions.k8s.io/nodefeatures.nfd.k8s-sigs.io created
  serviceaccount/nfd-gc created
  serviceaccount/nfd-master created
  serviceaccount/nfd-worker created
  role.rbac.authorization.k8s.io/nfd-worker created
  clusterrole.rbac.authorization.k8s.io/nfd-gc created
  clusterrole.rbac.authorization.k8s.io/nfd-master created
  rolebinding.rbac.authorization.k8s.io/nfd-worker created
  clusterrolebinding.rbac.authorization.k8s.io/nfd-gc created
  clusterrolebinding.rbac.authorization.k8s.io/nfd-master created
  configmap/nfd-master-conf-9mfc26f2tc created
  configmap/nfd-worker-conf-c2mbm9t788 created
  deployment.apps/nfd-gc created
  deployment.apps/nfd-master created
  daemonset.apps/nfd-worker created

$ kubectl -n node-feature-discovery get all
  NAME                              READY   STATUS    RESTARTS   AGE
  pod/nfd-gc-565fc85d9b-94jpj       1/1     Running   0          18s
  pod/nfd-master-6796d89d7b-qccrq   1/1     Running   0          18s
  pod/nfd-worker-nwdp6              1/1     Running   0          18s
...

$ kubectl get no -o json | jq ".items[].metadata.labels"
  {
    "kubernetes.io/arch": "amd64",
    "kubernetes.io/os": "linux",
    "feature.node.kubernetes.io/cpu-cpuid.ADX": "true",
    "feature.node.kubernetes.io/cpu-cpuid.AESNI": "true",
...

# 查看组件状态
root@node1:/usr/local# kubectl -n node-feature-discovery get pod
NAME                         READY   STATUS    RESTARTS   AGE
nfd-gc-59bbc7f68d-lrwd6      1/1     Running   0          25m
nfd-master-78d76fb47-nczvs   1/1     Running   0          25m
nfd-worker-bgr5h             1/1     Running   0          25m
nfd-worker-n28r2             1/1     Running   0          25m
nfd-worker-wlk4h             1/1     Running   0          25m

# 查看组件日志，可以看到nfd-worker组件默认每隔一分钟检测一次节点特性。
root@node1:/usr/local# kubectl logs -f -n=node-feature-discovery nfd-worker-wlk4h 
I0409 07:23:18.568325       1 nfd-worker.go:314] "Node Feature Discovery Worker" version="v0.17.2" nodeName="node2" namespace="node-feature-discovery"
I0409 07:23:18.568714       1 nfd-worker.go:519] "configuration file parsed" path="/etc/kubernetes/node-feature-discovery/nfd-worker.conf"
I0409 07:23:18.582145       1 nfd-worker.go:554] "configuration successfully updated" configuration={"Core":{"Klog":{},"LabelWhiteList":"","NoPublish":false,"NoOwnerRefs":false,"FeatureSources":["all"],"Sources":null,"LabelSources":["all"],"SleepInterval":{"Duration":60000000000}},"Sources":{"cpu":{"cpuid":{"attributeBlacklist":["AVX10","BMI1","BMI2","CLMUL","CMOV","CX16","ERMS","F16C","HTT","LZCNT","MMX","MMXEXT","NX","POPCNT","RDRAND","RDSEED","RDTSCP","SGX","SGXLC","SSE","SSE2","SSE3","SSE4","SSE42","SSSE3","TDX_GUEST"]}},"custom":[],"fake":{"labels":{"fakefeature1":"true","fakefeature2":"true","fakefeature3":"true"},"flagFeatures":["flag_1","flag_2","flag_3"],"attributeFeatures":{"attr_1":"true","attr_2":"false","attr_3":"10"},"instanceFeatures":[{"attr_1":"true","attr_2":"false","attr_3":"10","attr_4":"foobar","name":"instance_1"},{"attr_1":"true","attr_2":"true","attr_3":"100","name":"instance_2"},{"name":"instance_3"}]},"kernel":{"KconfigFile":"","configOpts":["NO_HZ","NO_HZ_IDLE","NO_HZ_FULL","PREEMPT"]},"local":{},"pci":{"deviceClassWhitelist":["03","0b40","12"],"deviceLabelFields":["class","vendor"]},"usb":{"deviceClassWhitelist":["0e","ef","fe","ff"],"deviceLabelFields":["class","vendor","device"]}}}
I0409 07:23:18.582453       1 metrics.go:44] "metrics server starting" port=":8081"
I0409 07:23:18.969370       1 nfd-worker.go:564] "starting feature discovery..."
I0409 07:23:18.969915       1 nfd-worker.go:577] "feature discovery completed"
I0409 07:23:18.995248       1 nfd-worker.go:672] "creating NodeFeature object" nodefeature="node2"
I0409 07:23:19.769563       1 component.go:34] [core][Server #1]Server created
I0409 07:23:19.769653       1 nfd-worker.go:228] "gRPC health server serving" port=8082
I0409 07:23:19.769798       1 component.go:34] [core][Server #1 ListenSocket #2]ListenSocket created
I0409 07:24:18.863911       1 nfd-worker.go:564] "starting feature discovery..."
I0409 07:24:18.864472       1 nfd-worker.go:577] "feature discovery completed"
I0409 07:25:18.863431       1 nfd-worker.go:564] "starting feature discovery..."
I0409 07:25:18.863947       1 nfd-worker.go:577] "feature discovery completed"

# 可以看到nfd-master组件启动后默认第一分钟相应地修改 Node 资源对象（标签、注解），之后是每隔一个小时修改一次 Node 资源对象（标签、注解），也就是说如果一个小时以内用户手动误修改node资源特性信息（标签、注解），最多需要一个小时nfd-master组件才自动更正node资源特性信息。
root@node1:/usr/local# kubectl logs -n=node-feature-discovery nfd-master-78d76fb47-nczvs 
I0409 07:23:01.705165       1 nfd-master.go:274] "Node Feature Discovery Master" version="v0.17.2" nodeName="node3" namespace="node-feature-discovery"
I0409 07:23:01.705403       1 nfd-master.go:1220] "configuration file parsed" path="/etc/kubernetes/node-feature-discovery/nfd-master.conf"
I0409 07:23:01.706106       1 nfd-master.go:1265] "configuration successfully updated" configuration=<
        AutoDefaultNs: true
        DenyLabelNs: {}
        EnableTaints: false
        ExtraLabelNs: {}
        Klog: {}
        LabelWhiteList: null
        LeaderElection:
          LeaseDuration:
            Duration: 15000000000
          RenewDeadline:
            Duration: 10000000000
          RetryPeriod:
            Duration: 2000000000
        NfdApiParallelism: 10
        NoPublish: false
        Restrictions:
          AllowOverwrite: true
          DenyNodeFeatureLabels: false
          DisableAnnotations: false
          DisableExtendedResources: false
          DisableLabels: false
          NodeFeatureNamespaceSelector: null
        ResyncPeriod:
          Duration: 3600000000000
 >
I0409 07:23:01.706262       1 nfd-master.go:1349] "starting the nfd api controller"
I0409 07:23:02.007421       1 nfd-api-controller.go:202] "informer caches synced" duration="300.842018ms"
I0409 07:23:02.007502       1 updater-pool.go:143] "starting the NFD master updater pool" parallelism=10
I0409 07:23:02.012984       1 metrics.go:44] "metrics server starting" port=":8081"
I0409 07:23:02.013294       1 component.go:34] [core][Server #1]Server created
I0409 07:23:02.013348       1 nfd-master.go:362] "gRPC health server serving" port=8082
I0409 07:23:02.013455       1 component.go:34] [core][Server #1 ListenSocket #2]ListenSocket created
I0409 07:23:03.013215       1 nfd-master.go:625] "will process all nodes in the cluster"
I0409 07:23:03.093033       1 nfd-master.go:1122] "node updated" nodeName="node2"
I0409 07:23:05.125124       1 nfd-master.go:1122] "node updated" nodeName="node3"
I0409 07:23:51.107306       1 nfd-master.go:1122] "node updated" nodeName="node1"

# 查看节点特性信息，可以看到NFD组件已经把节点特性信息维护到了节点标签、注解上，其中标签前缀默认为 feature.node.kubernetes.io/。
root@node1:/usr/local# kubectl describe node node1 
Name:               node1
Roles:              control-plane
Labels:             beta.kubernetes.io/arch=amd64
                    beta.kubernetes.io/os=linux
                    feature.node.kubernetes.io/cpu-cpuid.AESNI=true
                    feature.node.kubernetes.io/cpu-cpuid.AVX=true
                    feature.node.kubernetes.io/cpu-cpuid.CMPXCHG8=true
                    feature.node.kubernetes.io/cpu-cpuid.FLUSH_L1D=true
                    feature.node.kubernetes.io/cpu-cpuid.FXSR=true
                    feature.node.kubernetes.io/cpu-cpuid.FXSROPT=true
                    feature.node.kubernetes.io/cpu-cpuid.IBPB=true
                    feature.node.kubernetes.io/cpu-cpuid.LAHF=true
                    feature.node.kubernetes.io/cpu-cpuid.MD_CLEAR=true
                    feature.node.kubernetes.io/cpu-cpuid.OSXSAVE=true
                    feature.node.kubernetes.io/cpu-cpuid.SPEC_CTRL_SSBD=true
                    feature.node.kubernetes.io/cpu-cpuid.STIBP=true
                    feature.node.kubernetes.io/cpu-cpuid.SYSCALL=true
                    feature.node.kubernetes.io/cpu-cpuid.SYSEE=true
                    feature.node.kubernetes.io/cpu-cpuid.VMX=true
                    feature.node.kubernetes.io/cpu-cpuid.X87=true
                    feature.node.kubernetes.io/cpu-cpuid.XSAVE=true
                    feature.node.kubernetes.io/cpu-cpuid.XSAVEOPT=true
                    feature.node.kubernetes.io/cpu-cstate.enabled=true
                    feature.node.kubernetes.io/cpu-hardware_multithreading=false
                    feature.node.kubernetes.io/cpu-model.family=6
                    feature.node.kubernetes.io/cpu-model.id=62
                    feature.node.kubernetes.io/cpu-model.vendor_id=Intel
                    feature.node.kubernetes.io/cpu-pstate.status=passive
                    feature.node.kubernetes.io/cpu-pstate.turbo=false
                    feature.node.kubernetes.io/kernel-config.NO_HZ=true
                    feature.node.kubernetes.io/kernel-config.NO_HZ_IDLE=true
                    feature.node.kubernetes.io/kernel-version.full=5.15.0-134-generic
                    feature.node.kubernetes.io/kernel-version.major=5
                    feature.node.kubernetes.io/kernel-version.minor=15
                    feature.node.kubernetes.io/kernel-version.revision=0
                    feature.node.kubernetes.io/memory-numa=true
                    feature.node.kubernetes.io/network-sriov.capable=true
                    feature.node.kubernetes.io/pci-0300_18ca.present=true
                    feature.node.kubernetes.io/system-os_release.ID=ubuntu
                    feature.node.kubernetes.io/system-os_release.VERSION_ID=22.04
                    feature.node.kubernetes.io/system-os_release.VERSION_ID.major=22
                    feature.node.kubernetes.io/system-os_release.VERSION_ID.minor=04
                    kubernetes.io/arch=amd64
                    kubernetes.io/hostname=node1
                    kubernetes.io/os=linux
                    node-role.kubernetes.io/control-plane=
                    node.kubernetes.io/exclude-from-external-load-balancers=
Annotations:        flannel.alpha.coreos.com/backend-data: {"VNI":1,"VtepMAC":"fe:f6:f9:5f:8c:d3"}

Node Feature Discovery（NFD）组件的主要应用场景是在Kubernetes集群中提供更智能的节点调度。以下是一些NFD的常见应用场景：
1.智能节点调度：NFD可以帮助Kubernetes调度器更好地了解节点的特性和资源，从而更智能地选择最适合运行特定工作负载的节点。例如，如果某个Pod需要较强的GPU支持，调度器可以利用NFD标签来选择具有适当GPU型号的节点。
2.资源约束和优化：通过将节点的特性能力以标签的形式暴露给Kubernetes调度器，集群管理员可以更好地理解和利用集群中节点的资源情况，从而更好地进行资源约束和优化。
3.硬件感知的工作负载调度：对于特定的工作负载，可能需要特定类型或配置的硬件。NFD可以使调度器能够更加智能地选择具有适当硬件特性的节点来运行这些工作负载。
4.集群扩展性和性能：通过更智能地分配工作负载到节点，NFD可以提高集群的整体性能和效率。它可以帮助避免资源浪费，并确保工作负载能够充分利用可用的硬件资源。
5.集群自动化：NFD可以集成到自动化流程中，例如自动化部署或缩放工作负载。通过使用NFD，自动化系统可以更好地了解节点的特性和资源，从而更好地执行相应的操作。

Node Feature Discovery（NFD）可以帮助提高Kubernetes集群的智能程度，使其能够更好地适应各种类型的工作负载和节点特性，从而提高集群的性能、可靠性和效率。
如果 Kubernetes 集群需要根据节点的硬件特性进行智能调度或者对节点的硬件资源进行感知和利用，那么安装 Node Feature Discovery（NFD）是有必要的。然而，如果集群中的节点都具有相似的硬件配置，且不需要考虑硬件资源的差异，那么不需要安装 NFD。
```
![](IMAGE-1\MLOps-624219-20240314162752542-1472069471.png)

参考资料：
Kubernetes集群部署Node Feature Discovery组件用于检测集群节点特性    https://www.cnblogs.com/zhangmingcheng/p/18072751




# 四. Nvidia的GPU和CUDA分析工具NVIDIA NVIDIA-smi、NVIDIA Nsight Compute、NVIDIA Nsight Systems

### 5.NVIDIA NVIDIA-smi

**NVIDIA-smi**：这是NVIDIA提供的命令行工具，用于监控GPU的实时状态。它能够显示GPU利用率、显存使用情况、温度、功耗等关键指标。
**DCGM（Data Center GPU Manager）**：对于数据中心环境，DCGM提供了更高级的监控和管理功能，支持大规模GPU集群的监控和自动化管理。



**NVIDIA-smi**：

```
GPU利用率:
指标含义：GPU利用率表示GPU在单位时间内实际用于计算的时间比例。高利用率意味着GPU资源被充分利用，而低利用率可能表示资源浪费或计算任务未充分利用GPU的并行能力。
nvidia-smi --query-gpu=utilization.gpu --format=csv

显存利用率:
指标含义：显存利用率反映了GPU显存被占用的情况。如果显存利用率过高，可能会导致GPU频繁进行内存交换，降低计算性能。
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

计算吞吐量:
指标含义：计算吞吐量表示GPU在单位时间内能够完成的计算量。例如，在图像分类任务中，计算吞吐量可以表示为每秒处理的图像数量。
监控方法：可以通过AI框架（如TensorFlow、PyTorch）提供的日志或性能分析工具来测量计算吞吐量。

指标含义：GPU在高负载运行时会产生大量热量。监控GPU温度可以防止GPU过热而损坏。一般来说，GPU的温度应保持在安全范围内（通常在30 - 50摄氏度之间）。
nvidia-smi --query-gpu=temperature.gpu --format=csv

指标含义：GPU的功耗是一个重要的监控指标。一方面，要确保GPU的功耗在硬件允许的范围内，避免电源供应不足导致的硬件故障。另一方面，从能源成本角度考虑，合理控制GPU功耗也很重要。
nvidia-smi --query-gpu=power.draw --format=csv

实时监控GPU状态:
说明：watch命令用于实时监控，-n 1表示每秒刷新一次。nvidia-smi命令显示GPU的实时状态，包括利用率、显存使用情况、温度和功耗等。
watch -n 1 "nvidia-smi"

详细监控GPU性能:
说明：--query-gpu参数用于指定要查询的指标，--format=csv输出为CSV格式，方便后续分析。-l 1表示每秒记录一次数据。
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,power.draw,temperature.gpu --format=csv -l 5

监控特定GPU:
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -i 0

使用DCGM进行高级监控:
说明：dcgmi discovery -l列出所有可用的GPU和它们的状态。dcgmi dmon -e 1001 -d 1启动监控，-e 1001表示监控所有指标，-d 1表示每秒记录一次数据。
dcgmi discovery -l
dcgmi dmon -e 1001 -d 1

```



### 2.6.NVIDIA Nsight Compute

```
Nsight系列工具中的一个组件，专门用于CUDA核函数的性能分析，它是更接近内核的分析。它允许开发人员对 CUDA 核函数进行详细的性能分析，包括核函数的时间分布、内存访问模式、并行性、指令分发等。Nsight Compute提供了许多有用的数据和图形化的界面，帮助开发人员深入理解和优化核函数的性能。
ncu命令主要分析kernel内部的带宽、活跃线程数、warp利用率等。
地址：https://developer.nvidia.com/nsight-compute
DOC：https://developer.nvidia.com/tools-overview/nsight-compute/get-started

# nsight-compute包含在cuda的目录里。
/usr/local/cuda-12.4/nsight-compute-2024.1.1/

比较常用的分析：
①核函数的roofline分析，从而知道核函数是计算密集型还是访存密集型；
②occupancy analysis：对核函数的各个指标进行估算一个warp的占有率的变化；
③memory bindwidth analysis 针对核函数中对各个memory的数据传输带宽进行分析，可以比较好的理解memory架构；
④shared memory analysis：针对核函数中对shared memory访问以及使用效率进行分析；

/usr/local/cuda-12.4/nsight-compute-2024.1.1/
# 详细介绍。
https://docs.nvidia.com/nsight-compute/2024.3/CustomizationGuide/index.html

带宽查看有两个指标，分别是global memory的带宽和dram的带宽，global memory的带宽指标其实指的是L2Cache和L1Cache到SM的带宽，因为SM寻找数据会先去Cache寻找，Cache找不到再去GPU的DRAM中，所以有的时候会发现global memory的带宽会高于英伟达给的带宽参数。而DRAM带宽就是对应英伟达官方给的带宽。

命令行工具是/usr/local/cuda-12.4/nsight-compute-2024.1.1目录下的：ncu ncu-ui

# 查看Nsight-Compute支持的sections
ncu --list-sections

# 获取所有的metrics
ncu --set full --export ncu_report -f ./sample_2

# 获取指定section
ncu --section ComputeWorkloadAnalysis --print-details all ./sample_2

# 获取指定section和特定的metrics
ncu --section WarpStateStats --metrics smsp__pcsamp_sample_count,group:smsp__pcsamp_warp_stall_reasons,group:smsp__pcsamp_warp_stall_reasons_not_issued ./sample_2

ncu --metrics group:smsp__pcsamp_warp_stall_reasons ./sample_2
ncu --metrics group:smsp__pcsamp_warp_stall_reasons_not_issued ./sample_2
ncu --metrics group:memory__dram_table ./sample_2

ncu --metrics gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,breakdown:gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed  ./sample_2
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,breakdown:sm__throughput.avg.pct_of_peak_sustained_elapsed ./sample_2
ncu --metrics l1tex__lsuin_requests,l1tex__f_wavefronts,dram__sectors,dramc__sectors,fbpa__dram_sectors,l1tex__data_bank_reads,l1tex__data_pipe_lsu_wavefronts,l1tex__lsuin_requests,l1tex__t_bytes_lookup_miss  ./sample_2

ncu --metrics sass__inst_executed_per_opcode,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active --target-processes all --export ncu_report -f ./sample_2

# 查看instances的数据
ncu -i ncu_report.ncu-rep --print-details all --print-units auto --print-metric-instances values

参考资料：
NVIDIA性能分析工具nsight-compute入门    https://zhuanlan.zhihu.com/p/662012270
NsightComputeProfiling入门    https://blog.csdn.net/m0_61864577/article/details/140618800
```



### 2.7.NVIDIA Nsight Systems

```
系统级的性能分析工具，用于分析和优化整个CUDA应用程序或系统的性能。它可以提供对应用程序整体性能的全面见解，以及考察GPU活动、内存使用、线程间通信等方面的详细信息，它提供了可视化界面和统计数据，开发人员可以使用它来发现性能瓶颈、调整应用程序的配置，以及提高整体性能。
nsys命令主要分析api级别的性能时间等。
地址：https://developer.nvidia.com/nsight-systems
DOC：https://developer.nvidia.com/nsight-systems/get-started

# nsight-compute包含在cuda的目录里。
/usr/local/cuda-12.4/nsight-systems-2023.4.4/
命令行工具为/usr/local/cuda-12.4/nsight-systems-2023.4.4/bin下的nsys，nsys-ui。

nsys 提供了强大的命令行界面（CLI），方便用户进行各种性能分析操作。以下是一些最常用的命令及其功能：

nsys profile [options] [application] [application args]: 这是最核心的命令，用于启动应用程序并捕获其性能数据。

--trace=<trace>: 通过这个选项，你可以指定要跟踪的 API 或事件类型，例如 cuda（跟踪 CUDA API 和内核）、cudart（跟踪 CUDA 运行时 API）、osrt（跟踪操作系统运行时 API）、opengl、vulkan 等。你可以使用逗号分隔多个跟踪类型，例如 --trace=cuda,osrt,vulkan。
-o <filename>: 使用此选项指定输出报告文件的名称，通常以 .qdrep 格式保存。例如，-o my_report.qdrep。
--duration=<seconds>: 设置性能分析的持续时间，单位为秒。例如，--duration=10 将会分析应用程序运行的 10 秒。
--delay=<seconds>: 设置开始性能分析前的延迟时间，单位为秒。这在需要等待应用程序启动完成后再开始分析时非常有用。
--gpu-metrics-device=<device_id>: 如果你的系统中有多个 GPU，可以使用此选项指定要收集 GPU 指标的设备 ID。
--cudabacktrace=all|api|kernel|none: 控制 CUDA 回溯信息的收集级别，有助于更深入地了解 CUDA API 调用和内核执行的上下文。
想要了解更多选项，请随时使用 nsys profile --help 命令查看完整的帮助文档。
nsys launch [options] [application] [application args]: 这个命令用于启动应用程序，并使其处于等待性能分析器连接的状态。这在你需要从另一个进程或机器上连接 nsys 进行分析时非常有用。

nsys start [options]: 启动一个新的性能分析会话。通常会与 nsys stop 命令配合使用，用于在应用程序运行的特定时间段内进行性能分析。
--trace=<trace>: 同样用于指定要跟踪的 API 或事件类型。
--output=<filename>: 指定输出报告文件的名称。

nsys stop: 停止当前正在运行的性能分析会话，并将收集到的性能数据保存到指定的文件中。
nsys cancel: 如果你想要放弃当前的性能分析会话，可以使用这个命令取消并丢弃已经收集到的数据。
nsys service: 启动 Nsight Systems 数据服务，这是 nsys 工具后台运行的一个重要组成部分。
nsys stats <filename>.qdrep: 这个命令可以从一个已经存在的 .qdrep 报告文件中生成各种统计信息，帮助你快速了解性能概况。
nsys status: 显示当前 nsys 的运行状态，例如是否有正在进行的性能分析会话。
nsys shutdown: 关闭所有与 nsys 相关的进程。
nsys sessions list: 列出当前所有活动的性能分析会话。
nsys export <filename>.qdrep --type=<format> -o <output_filename>: 将 .qdrep 报告文件导出为其他格式，例如 csv（逗号分隔值）、sqlite（SQLite 数据库）等，方便与其他工具进行数据交换和分析。
nsys analyze <filename>.qdrep: 分析报告文件，nsys 可能会识别出潜在的性能优化机会并给出建议。
nsys recipe <recipe_file>: 用于运行多节点分析的配方文件，这对于分析分布式应用程序非常有用。
nsys nvprof [nvprof options] [application] [application args]: 对于熟悉 NVIDIA 之前的性能分析工具 nvprof 的用户，nsys 提供了这个命令来尝试将 nvprof 的命令行选项转换为 nsys 的选项并执行分析，方便用户进行迁移。

要获取任何特定命令的更详细信息，只需在终端中运行 nsys --help <command> 即可。例如，要查看 nsys profile 命令的所有可用选项，可以执行 nsys --help profile。


比较常用的分析：
①对kernel执行和memory进行timeline分析，尝试优化,隐藏memory access或者删除冗长的memory access；多流调度，融合kernel减少kernel launch的overhead；CPU与GPU的overlapping；
②分析DRAM以及PCIe带宽的使用率，没有使用shared memory，那么DRAM brandwidth就没有变化，可以分析哪些带宽没有被充分利用；
③分析warp的占有率，从而可以知道一个SM中计算资源是否被充分利用；

NVIDIA Nsight Systems (nsys) 是一款功能强大的系统级性能分析工具，它通过提供全方位的性能数据和直观的可视化界面，帮助开发者深入了解应用程序在整个系统中的行为。掌握 nsys 的使用，能够让你更有效地识别性能瓶颈，优化资源利用率，最终提升应用程序的整体效率。

参考资料：
cuda学习日记(6) nsight system / nsight compute    https://zhuanlan.zhihu.com/p/640344249
NVIDIA Nsight Systems (nsys) 工具使用    https://www.cnblogs.com/menkeyi/p/18791669
```



KAI Scheduler：优化GPU资源分配的Kubernetes调度器

```
https://github.com/NVIDIA/KAI-Scheduler

项目核心功能/场景
KAI Scheduler 是一款强大的Kubernetes调度器，专注于优化AI和机器学习工作负载的GPU资源分配。

项目介绍
KAI Scheduler 设计用于管理大规模GPU集群，包括成千上万的节点和高吞吐量的工作负载。它特别适合于广泛和苛刻的环境。管理员可以利用KAI Scheduler动态地为Kubernetes集群中的工作负载分配GPU资源。

KAI Scheduler 支持整个AI生命周期，从需要最少资源的小型交互式任务到同一集群内的大型训练和推理任务。它确保了资源的最优分配，并在不同的消费者之间保持资源公平性。它还可以与其他已安装在集群上的调度器共同运行。

项目技术分析
KAI Scheduler 基于Kubernetes调度器，提供了一系列高级特性，用于优化GPU资源的调度和管理。以下是其技术特点：

批调度：确保一个组内的所有Pod要么同时被调度，要么一个都不调度。
装箱调度与扩散调度：通过最小化碎片化（装箱调度）或增加弹性和负载均衡（扩散调度）来优化节点使用。
工作负载优先级：在队列中有效地优先调度工作负载。
分层队列：使用两级队列层次结构管理工作负载，实现灵活的组织控制。
资源分配：为每个队列自定义配额、超配额权重、限制和优先级。
公平性策略：使用支配资源公平性（DRF）和跨队列的资源回收策略确保公平的资源分配。
工作负载合并：智能地重新分配运行中的工作负载，以减少碎片化和提高集群利用率。
弹性工作负载：在定义的最小和最大Pod数量范围内动态调整工作负载。
动态资源分配：通过Kubernetes ResourceClaims支持特定供应商的硬件资源（例如，NVIDIA或AMD的GPU）。
GPU共享：允许多个工作负载高效地共享一个或多个GPU，最大化资源利用率。
云和本地支持：完全兼容动态云基础设施（包括自动扩展器如Karpenter）以及静态本地部署。

项目技术应用场景
KAI Scheduler 适用于以下几种技术应用场景：
大型AI训练：对于需要大量GPU资源的大型AI训练任务，KAI Scheduler可以有效地管理和分配资源，确保训练任务的高效执行。
机器学习模型推理：在模型推理阶段，KAI Scheduler能够动态调整资源，以满足不同负载的需求，提高资源利用率。
多云和混合云环境：无论是在云环境还是本地部署中，KAI Scheduler都能提供一致的调度策略，确保资源的高效使用。

项目特点
高度可扩展性：KAI Scheduler 能够处理大规模的GPU集群，确保在高负载情况下依然能够高效调度资源。
灵活性：通过分层队列和工作负载优先级，管理员可以根据需要灵活管理资源分配。
资源优化：通过工作负载合并和弹性工作负载特性，KAI Scheduler 能够最大化资源利用率。
兼容性：支持多种部署环境，包括动态云基础设施和静态本地部署。
总结来说，KAI Scheduler 是一个针对GPU资源优化调度的高效Kubernetes调度器，适用于多种规模和复杂度的AI和机器学习工作负载。通过其先进的技术特性和灵活的调度策略，它为管理员提供了一个强大的工具来管理和优化GPU资源。无论是大型训练任务还是多云环境中的模型推理，KAI Scheduler 都能够提供出色的性能和效率。
```




