
# metrics-server

https://github.com/kubernetes-sigs/metrics-server/tree/release-0.8
```
可以通过kubectl get apiservices命令查询集群中的APIService


官网的部署命令是：
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.8.0/components.yaml

但是，需要下载yaml文件，修改一下，禁用证书验证。
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
        image: registry.k8s.io/metrics-server/metrics-server:v0.8.0
################################

通过以下命令测试Metrics Server的API功能
kubectl get --raw /apis/metrics.k8s.io/v1beta1/nodes

```

Metrics Server与其他工具的对比
以下表格展示了Metrics Server与其他监控工具的对比：

工具	功能	数据存储	可视化支持
Metrics Server	集群资源度量API	不支持	不支持
Prometheus	完整监控和报警系统	支持	支持
Grafana	数据可视化工具	不支持	支持
