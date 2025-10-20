
# metrics-server

https://github.com/kubernetes-sigs/metrics-server/tree/release-0.8
```
可以通过kubectl get apiservices命令查询集群中的APIService


官网的部署命令是：
wget https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.8.0/components.yaml
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.8.0/components.yaml


2. Kubernetes Metrics Server
https://github.com/kubernetes-sigs/metrics-server

官网的部署命令是：
wget https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.8.0/components.yaml
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.8.0/components.yaml

# 修改yaml文件中的镜像地址：
需要下载yaml文件，修改一下，禁用证书验证。
components.yaml
################################
  template:
    metadata:
      labels:
        k8s-app: metrics-server
    spec:
      containers:
      - args:
        - --kubelet-insecure-tls    # 添加这一行
        - --cert-dir=/tmp
        - --secure-port=10250
        - --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname
        - --kubelet-use-node-status-port
        - --metric-resolution=15s
        - --kubelet-preferred-address-types=InternalIP,Hostname,InternalDNS,ExternalDNS,ExternalIP   # added
        image: m.daocloud.io/registry.k8s.io/metrics-server/metrics-server:v0.8.0    #registry.k8s.io/metrics-server/metrics-server:v0.8.0
################################
registry.k8s.io/metrics-server/metrics-server:v0.8.0
m.daocloud.io/registry.k8s.io/metrics-server/metrics-server:v0.8.0


1. 基础状态检查命令
检查Metrics Server部署状态
# 检查Metrics Server Pod状态
kubectl get pods -n kube-system -l k8s-app=metrics-server

# 检查Metrics Server部署
kubectl get deployment -n kube-system metrics-server

# 检查Metrics Server服务
kubectl get service -n kube-system metrics-server

# 查看详细部署信息
kubectl describe deployment -n kube-system metrics-server
检查API资源是否注册
# 检查Metrics API是否可用
kubectl get --raw /apis/metrics.k8s.io/v1beta1 | jq .

# 或者使用更简洁的方式
kubectl get apiservices | grep metrics
kubectl get --raw /apis/metrics.k8s.io/v1beta1/nodes

2. 资源指标查询命令
查看节点资源指标
# 查看所有节点的资源使用情况
kubectl top nodes

# 查看特定节点的资源使用
kubectl top node <node-name>

# 以宽格式显示更多信息
kubectl top nodes --use-protocol-buffers
查看Pod资源指标
# 查看所有命名空间的Pod资源使用
kubectl top pods --all-namespaces

# 查看特定命名空间的Pod资源使用
kubectl top pods -n <namespace>

# 查看特定Pod的资源使用
kubectl top pod <pod-name> -n <namespace>

# 包含标签信息
kubectl top pods -l app=my-app

3. 诊断和故障排除命令
检查Metrics Server日志
# 查看Metrics Server Pod日志
kubectl logs -n kube-system -l k8s-app=metrics-server

# 查看特定Pod的详细日志
kubectl logs -n kube-system deployment/metrics-server

# 实时查看日志
kubectl logs -n kube-system -l k8s-app=metrics-server -f

诊断API连接问题
# 检查API服务状态
kubectl get apiservice v1beta1.metrics.k8s.io -o yaml

# 检查端点状态
kubectl get endpoints -n kube-system metrics-server

# 检查服务发现
# 检查服务证书
kubectl get --raw /api/v1/namespaces/kube-system/services/https:metrics-server:/proxy/healthz  # 执行后报错：Error from server (ServiceUnavailable): no endpoints available for service "https:metrics-server:"

4. 高级测试命令
直接访问Metrics API
# 获取所有节点的metrics数据
kubectl get --raw /apis/metrics.k8s.io/v1beta1/nodes | jq .

# 获取所有pods的metrics数据
kubectl get --raw /apis/metrics.k8s.io/v1beta1/pods | jq .

# 获取特定命名空间的pods metrics
kubectl get --raw /apis/metrics.k8s.io/v1beta1/namespaces/kube-system/pods | jq .

# 检查Pod安全配置
kubectl get pod -n kube-system -l k8s-app=metrics-server -o yaml | grep -A5 securityContext

```



## Metrics Server与其他工具的对比

以下表格展示了Metrics Server与其他监控工具的对比：

| 工具           | 功能               | 数据存储 | 可视化支持 |
| -------------- | ------------------ | -------- | ---------- |
| Metrics Server | 集群资源度量API    | 不支持   | 不支持     |
| Prometheus     | 完整监控和报警系统 | 支持     | 支持       |
| Grafana        | 数据可视化工具     | 不支持   | 支持       |
