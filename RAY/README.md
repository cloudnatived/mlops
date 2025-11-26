


```


8. 分布式计算集群Ray
RAY
docker 方式部署ray集群，未部署成功，版本太高。：
docker pull hub.rat.dev/rayproject/ray-ml:nightly-py39-gpu # 镜像里的cuda版本是：CUDA Version 12.1.1，未部署成功，版本太高。
docker pull hub.rat.dev/rayproject/ray:nightly-py39-cu128  # 镜像里的cuda版本是：CUDA Version 12.8.1，未部署成功，版本太高。
docker pull hub.rat.dev/rayproject/ray:2.37.0

# 启动 Head 节点
# 在GPU的主机上，运行一个cuda:11.8.0-cudnn8-runtime-ubuntu22.04容器，在这个容器里安装vllm，ray，并做为ray的Head节点。
# 因为需要python低于Python 3.11，ubuntu22.04的python版本是3.10.12，在CPU节点上启动一个ubuntu22.04用来安装ray。
docker run -it -d --cap-add SYS_ADMIN --net=host --pid=host --ipc=host --privileged nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 每个节点上启动一个镜像，用来部署ray，如果需要挂载vllm模型目录，每个准备作为ray的节点的容器，都需要预先就挂载上
docker run -it -d --gpus all --cap-add SYS_ADMIN --net=host --pid=host --ipc=host --privileged -v /Data/Qwen/Qwen2.5-1.5B:/model nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
docker run -it -d --gpus all --cap-add SYS_ADMIN --net=host --pid=host --ipc=host --privileged -v /Data/Qwen/Qwen2.5-14B:/model nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

docker run --rm --cap-add SYS_ADMIN --net=host --pid=host --ipc=host --privileged hub.rat.dev/rayproject/ray:2.37.0


# 下载了54个python包
# 把安装包都下载到了/Data/IMAGES/whl/目录中
-rw-r--r-- 1 root root     31711 Sep  9 02:23 aioprometheus-23.12.0-py3-none-any.whl
-rw-r--r-- 1 root root     13643 Sep  9 02:23 annotated_types-0.7.0-py3-none-any.whl
-rw-r--r-- 1 root root    107213 Sep  9 02:23 anyio-4.10.0-py3-none-any.whl
-rw-r--r-- 1 root root     16674 Sep  9 02:23 exceptiongroup-1.3.0-py3-none-any.whl
-rw-r--r-- 1 root root     95631 Sep  9 02:23 fastapi-0.116.1-py3-none-any.whl
-rw-r--r-- 1 root root    199289 Sep  9 02:23 fsspec-2025.9.0-py3-none-any.whl
-rw-r--r-- 1 root root     37515 Sep  9 02:23 h11-0.16.0-py3-none-any.whl
-rw-r--r-- 1 root root   3186789 Sep  9 02:23 hf_xet-1.1.9-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
-rw-r--r-- 1 root root    442148 Sep  9 02:23 httptools-0.6.4-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl
-rw-r--r-- 1 root root    561452 Sep  9 02:23 huggingface_hub-0.34.4-py3-none-any.whl
-rw-r--r-- 1 root root    134899 Sep  9 02:23 jinja2-3.1.6-py3-none-any.whl
-rw-r--r-- 1 root root     20866 Sep  9 02:23 MarkupSafe-3.0.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
-rw-r--r-- 1 root root    536198 Sep  9 02:23 mpmath-1.3.0-py3-none-any.whl
-rw-r--r-- 1 root root   1723263 Sep  9 02:23 networkx-3.4.2-py3-none-any.whl
-rw-r--r-- 1 root root    180716 Sep  9 02:23 ninja-1.13.0-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
-rw-r--r-- 1 root root  16801050 Sep  9 02:23 numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
-rw-r--r-- 1 root root 410594774 Sep  9 02:23 nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl
-rw-r--r-- 1 root root 581242350 Sep  9 01:52 nvidia_cublas_cu12-12.9.1.4-py3-none-manylinux_2_27_x86_64.whl
-rw-r--r-- 1 root root  14109015 Sep  9 02:23 nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl
-rw-r--r-- 1 root root  23671734 Sep  9 02:23 nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl
-rw-r--r-- 1 root root    823596 Sep  9 02:23 nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl
-rw-r--r-- 1 root root 731725872 Sep  9 02:23 nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl
-rw-r--r-- 1 root root 121635161 Sep  9 02:23 nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl
-rw-r--r-- 1 root root  56467784 Sep  9 02:23 nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl
-rw-r--r-- 1 root root 124161928 Sep  9 02:23 nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl
-rw-r--r-- 1 root root 195958278 Sep  9 02:23 nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl
-rw-r--r-- 1 root root 366465088 Sep  9 01:48 nvidia_cusparse_cu12-12.5.10.65-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
-rw-r--r-- 1 root root 366465088 Sep  9 08:56 nvidia_cusparse_cu12-12.5.10.65-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.current
-rw-r--r-- 1 root root 209797524 Sep  9 02:23 nvidia_nccl_cu12-2.18.1-py3-none-manylinux1_x86_64.whl
-rw-r--r-- 1 root root     99138 Sep  9 02:23 nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl
-rw-r--r-- 1 root root    132966 Sep  9 02:23 orjson-3.11.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
-rw-r--r-- 1 root root    277986 Sep  9 02:23 psutil-7.0.0-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl
-rw-r--r-- 1 root root    444782 Sep  9 02:23 pydantic-2.11.7-py3-none-any.whl
-rw-r--r-- 1 root root   2005652 Sep  9 02:23 pydantic_core-2.33.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
-rw-r--r-- 1 root root     53135 Sep  9 02:23 pynvml-11.5.0-py3-none-any.whl
-rw-r--r-- 1 root root     20556 Sep  9 02:23 python_dotenv-1.1.1-py3-none-any.whl
-rw-r--r-- 1 root root  69935694 Sep  9 02:23 ray-2.49.1-cp310-cp310-manylinux2014_x86_64.whl
-rw-r--r-- 1 root root    789931 Sep  9 02:23 regex-2025.9.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
-rw-r--r-- 1 root root    485835 Sep  9 02:23 safetensors-0.6.2-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
-rw-r--r-- 1 root root   1387926 Sep  9 02:23 sentencepiece-0.2.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
-rw-r--r-- 1 root root     10235 Sep  9 02:23 sniffio-1.3.1-py3-none-any.whl
-rw-r--r-- 1 root root     72991 Sep  9 02:23 starlette-0.47.3-py3-none-any.whl
-rw-r--r-- 1 root root   6299353 Sep  9 02:23 sympy-1.14.0-py3-none-any.whl
-rw-r--r-- 1 root root   3345585 Sep  9 02:23 tokenizers-0.22.0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
-rw-r--r-- 1 root root 670178687 Sep  9 02:23 torch-2.1.2-cp310-cp310-manylinux1_x86_64.whl
-rw-r--r-- 1 root root     78540 Sep  9 02:23 tqdm-4.67.1-py3-none-any.whl
-rw-r--r-- 1 root root  11608197 Sep  9 02:23 transformers-4.56.1-py3-none-any.whl
-rw-r--r-- 1 root root  89215103 Sep  9 02:23 triton-2.1.0-0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
-rw-r--r-- 1 root root     14552 Sep  9 02:23 typing_inspection-0.4.1-py3-none-any.whl
-rw-r--r-- 1 root root     66406 Sep  9 02:23 uvicorn-0.35.0-py3-none-any.whl
-rw-r--r-- 1 root root   3825126 Sep  9 02:23 uvloop-0.21.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
-rw-r--r-- 1 root root  38035860 Sep  9 02:23 vllm-0.3.0-cp310-cp310-manylinux1_x86_64.whl
-rw-r--r-- 1 root root    453148 Sep  9 02:23 watchfiles-1.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
-rw-r--r-- 1 root root    181631 Sep  9 02:23 websockets-15.0.1-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl
-rw-r--r-- 1 root root 212986727 Sep  9 02:23 xformers-0.0.23.post1-cp310-cp310-manylinux2014_x86_64.whl

# 在python安装文件的目录里执行
ls |grep whl|xargs -i pip3 install {} --no-dependencies

# 安装ray
pip3 install ray[default] --no-dependencies --find-links=/Data/IMAGES/whl
pip install "numpy<2" "transformers<4.40" "torch==2.1.2"
pip install "numpy<2" "transformers<4.40" "torch==2.1.2" --no-dependencies --find-links=/Data/IMAGES/whl

# ray的版本
ray --version 
ray, version 2.49.1

# 启动 Head 节点，在172.18.8.210节点上先启动ray，作为Head节点启动ray
ray stop # 先执行stop，确保ray处在停止状态
ray stop --force
ray start --head --node-ip-address=172.18.8.208 --port=6379 --dashboard-host=0.0.0.0 --include-dashboard=true --dashboard-port=8265 --num-cpus=4 --num-gpus=2

# 启动 Worker 节点，在172.18.8.208和172.18.8.209这两个GPU云主机上启动ray，作为work节点
ray start --address='172.18.8.210:6379' --node-ip-address=172.18.8.208 --num-cpus=4 --num-gpus=2
ray start --address='172.18.8.210:6379' --node-ip-address=172.18.8.209 --num-cpus=4 --num-gpus=2

# 验证集群状态​
# Head 节点上，查看集群节点
ray list nodes
ray status
# 预期输出应包含所有节点和资源

# RAY ，测试API连通性，在任意节点上执行
curl -v http://172.18.8.210:8265/api/v0/nodes

訓練任務是直接失敗，還是能夠（如果使用了 Ray 或其他彈性框架）檢測到節點丟失並嘗試恢復或繼續運行

```


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
Method 1: Helm (Recommended)                                              # 方法一 使用Helm部署
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
# Install both CRDs and KubeRay operator v1.3.0.
helm install kuberay-operator kuberay/kuberay-operator --version 1.3.0

Method 2: Kustomize                                                       # 方法二 本次部署使用 kubectl部署
# Install CRD and KubeRay operator.
kubectl create -k "github.com/ray-project/kuberay/ray-operator/config/default?ref=v1.3.0" 

Step 3: Validate Installation
kubectl get pods|grep ray                                                 # 检查kuberay-operator是否正常运行


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


