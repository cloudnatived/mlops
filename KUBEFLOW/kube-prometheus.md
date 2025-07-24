### kube-prometheus-0.15.0

```
主要参考文档：
https://www.cnblogs.com/niuben/p/18888238   主要是看的这个文档里的NFS。
https://cloud.tencent.com/developer/article/1780158


项目来自：
https://github.com/prometheus-operator/kube-prometheus

为什么要部署kube-prometheus，在物理机层面，在kubernetes层面，在istio层面，都有prometheus，部署kube-prometheus是因为其比较好的集成度和向上和向下的兼容。

创建了文件：
kube-prometheus-pv.yaml
kube-prometheus-storage-class.yaml
kube-prometheus-values.yaml 

修改了文件：
alertmanager-service.yaml
grafana-service.yaml
nodeExporter-service.yaml
prometheus-service.yaml

按照官网，执行部署的命令：
kubectl apply --server-side -f manifests/setup
kubectl wait \
	--for condition=Established \
	--all CustomResourceDefinition \
	--namespace=monitoring
kubectl apply -f manifests/



```
