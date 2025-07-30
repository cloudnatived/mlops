#### 2.4.5 基于 Kubernetes 和 Ray 进行分布式训练



Ray项目包含很多模块，包括实现基本分布式能力的Ray Core，进行数据处理的[Ray Data](https://zhida.zhihu.com/search?content_id=230301607&content_type=Article&match_order=1&q=Ray+Data&zhida_source=entity)，进行训练的[Ray Train](https://zhida.zhihu.com/search?content_id=230301607&content_type=Article&match_order=1&q=Ray+Train&zhida_source=entity)，超参数调整的[Ray Tune](https://zhida.zhihu.com/search?content_id=230301607&content_type=Article&match_order=1&q=Ray+Tune&zhida_source=entity)，实现推理的[Ray Serve](https://zhida.zhihu.com/search?content_id=230301607&content_type=Article&match_order=1&q=Ray+Serve&zhida_source=entity)，强化学习库[Ray RLlib](https://zhida.zhihu.com/search?content_id=230301607&content_type=Article&match_order=1&q=Ray+RLlib&zhida_source=entity)，以及集合了多种功能的上层机器学习API Ray AIR。

Ray 的三个核心概念：

- **Task（任务）**：通过 `@ray.remote` 装饰器，可以将普通 Python 函数转换为分布式任务，实现无状态的并行计算

- **Actor（角色）**：为分布式环境提供有状态计算的抽象，支持面向对象的并行编程模型。有状态的分布式对象

- **Object Store（对象存储）**：Ray 的分布式共享内存系统，实现高效的跨节点数据共享和传输

  

| Ray Core  |                                      |
| --------- | ------------------------------------ |
| Ray Core  | 实现基本分布式能力                   |
| Ray Data  | 数据处理                             |
| Ray Train |                                      |
| Ray Tune  | 用于自动化超参数调优                 |
| Ray Serve | 用于模型部署和服务                   |
| Ray RLlib | 是基于 Tune 开发的分布式强化学习框架 |
| Datasets  | 处理分布式数据                       |



Ray 官方文档的安装文档：https://docs.ray.io/en/latest/ray-overview/installation.html
Ray 中文参考文档：https://scale-py.godaai.org/index.html

在物理服务器上部署ray，直接pip install即可，注意不同安装命令会安装不同的库，

| Command                       | Installed components                                         |
| ----------------------------- | ------------------------------------------------------------ |
| pip install -U "ray"          | Core                                                         |
| pip install -U "ray[default]" | Core, Dashboard, Cluster Launcher                            |
| pip install -U "ray[data]"    | Core, Data                                                   |
| pip install -U "ray[train]"   | Core, Train                                                  |
| pip install -U "ray[tune]"    | Core, Tune                                                   |
| pip install -U "ray[serve]"   | Core, Dashboard, Cluster Launcher, Serve                     |
| pip install -U "ray[rllib]"   | Core, Tune, RLlib                                            |
| pip install -U "ray[air]"     | Core, Dashboard, Cluster Launcher, Data, Train, Tune, Serve  |
| pip install -U "ray[all]"     | Core, Dashboard, Cluster Launcher, Data, Train, Tune, Serve, RLlib |



​     

基于 docker 部署Ray：

```
[Ray-2.46.0](https://github.com/ray-project/ray/releases/tag/ray-2.46.0)   

docker pull rayproject/ray:2.46.0-py39-cpu 
819.38 MB

docker pull rayproject/ray:2.46.0-py39-cu124
5.22 GB

$ kubectl -n ray-system apply -f https://raw.githubusercontent.com/ray-project/kuberay/master/ray-operator/config/samples/ray-cluster.autoscaler.yaml

ray 的 docker hub地址：https://hub.docker.com/r/rayproject/ray/



```



使用 KubeRay Operator 基于 Kubernetes 部署Ray：

```
https://github.com/ray-project/kuberay/
https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/kuberay-operator-installation.html
中文文档：
https://docs.rayai.org.cn/en/latest/cluster/kubernetes/getting-started/kuberay-operator-installation.html


1. 部署 KubeRay Operator，本次部署版本v1.3.0
Step 1: Create a Kubernetes cluster                                       # 使用kind新建kubernetes集群，如果已有kubernetes集群，跳过此步骤
kind create cluster --image=kindest/node:v1.26.0

Step 2: Install KubeRay operator
Method 1: Helm (Recommended)                                              # 方法一
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
# Install both CRDs and KubeRay operator v1.3.0.
helm install kuberay-operator kuberay/kuberay-operator --version 1.3.0

Method 2: Kustomize                                                       # 方法二 本次部署使用 kubectl部署
# Install CRD and KubeRay operator.
kubectl create -k "github.com/ray-project/kuberay/ray-operator/config/default?ref=v1.3.0" 

Step 3: Validate Installation
kubectl get pods                                                          # 检查kuberay-operator是否正常运行


2. 部署 RayCluster
kubectl apply -f https://raw.githubusercontent.com/ray-project/kuberay/v1.3.0/ray-operator/config/samples/ray-cluster.sample.yaml

# 验证
kubectl get rayclusters
kubectl get pods --selector=ray.io/cluster=raycluster-kuberay


3. 部署 RayJob
kubectl apply -f https://raw.githubusercontent.com/ray-project/kuberay/v1.3.0/ray-operator/config/samples/ray-job.sample.yaml
kubectl apply -f https://raw.githubusercontent.com/ray-project/kuberay/v1.3.0/ray-operator/config/samples/ray-service.sample.yaml

# 验证 
kubectl get rayjob
kubectl get raycluster
kubectl get pods --sort-by='.metadata.creationTimestamp'
kubectl get rayjobs.ray.io rayjob-sample -o jsonpath='{.status.jobStatus}'
kubectl get rayjobs.ray.io rayjob-sample -o jsonpath='{.status.jobDeploymentStatus}'
kubectl logs -l=job-name=rayjob-sample

# Delete the RayJob
kubectl apply -f https://raw.githubusercontent.com/ray-project/kuberay/v1.3.0/ray-operator/config/samples/ray-job.shutdown.yaml
kubectl get rayjobs.ray.io rayjob-sample-shutdown -o jsonpath='{.status.jobDeploymentStatus}'
kubectl get rayjobs.ray.io rayjob-sample-shutdown -o jsonpath='{.status.jobStatus}'


4. 部署 ray-service
kubectl apply -f https://raw.githubusercontent.com/ray-project/kuberay/v1.3.0/ray-operator/config/samples/ray-service.sample.yaml

kubectl get rayservice
kubectl get raycluster
kubectl get pods -l=ray.io/is-ray-node=yes

kubectl get rayservice rayservice-sample -o json | jq -r '.status.conditions[] | select(.type=="Ready") | to_entries[] | "\(.key): \(.value)"'
kubectl get services -o json | jq -r '.items[].metadata.name'

# Verify the status of the Serve applications
kubectl port-forward svc/rayservice-sample-head-svc 8265:8265 > /dev/null &

# Send requests to the Serve applications by the Kubernetes serve service
kubectl run curl --image=radial/busyboxplus:curl --command -- tail -f /dev/null
kubectl exec curl -- curl -sS -X POST -H 'Content-Type: application/json' rayservice-sample-serve-svc:8000/calc/ -d '["MUL", 3]'
kubectl exec curl -- curl -sS -X POST -H 'Content-Type: application/json' rayservice-sample-serve-svc:8000/fruit/ -d '["MANGO", 2]'

Example

kubectl apply -f https://raw.githubusercontent.com/ray-project/ray/releases/2.0.0/doc/source/cluster/kubernetes/configs/xgboost-benchmark.yaml

# Train a PyTorch model on Fashion MNIST with CPUs on Kubernetes
curl -LO https://raw.githubusercontent.com/ray-project/kuberay/master/ray-operator/config/samples/pytorch-mnist/ray-job.pytorch-mnist.yaml

# Serve a MobileNet image classifier on Kubernetes
kubectl apply -f https://raw.githubusercontent.com/ray-project/kuberay/v1.3.0/ray-operator/config/samples/ray-service.mobilenet.yaml

```


