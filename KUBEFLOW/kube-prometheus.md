
3. kube-prometheus
### kube-prometheus-0.15.0
### kube-prometheus-0.16.0

```
主要参考文档：
https://www.cnblogs.com/niuben/p/18888238   主要是看的这个文档里的NFS。
https://cloud.tencent.com/developer/article/1780158


项目来自：
https://github.com/prometheus-operator/kube-prometheus

https://github.com/prometheus-operator/kube-prometheus/archive/refs/tags/v0.16.0.tar.gz

为什么要部署kube-prometheus，在物理机层面，在kubernetes层面，在istio层面，都有prometheus，部署kube-prometheus是因为其比较好的集成度和向上和向下的兼容。

创建文件：
kube-prometheus-pv.yaml
kube-prometheus-storage-class.yaml
kube-prometheus-values.yaml

修改了文件，把 ClusterIP 修改成 NodePort ：
alertmanager-service.yaml
grafana-service.yaml
prometheus-service.yaml

# 使用代理的镜像地址：
sed -i "s#image: #image: m.daocloud.io/#g" *
sed -i "s#m.daocloud.io/grafana/grafana:12.1.0#hub.rat.dev/grafana/grafana:12.1.0#g" grafana-deployment.yaml
kubectl apply -f grafana-deployment.yaml

按照官网，执行部署的命令：
kubectl apply --server-side -f manifests/setup

kubectl wait \
	--for condition=Established \
	--all CustomResourceDefinition \
	--namespace=monitoring

kubectl apply -f manifests/

# kubectl wait --for condition=Established --all CustomResourceDefinition --namespace=monitoring


# 如果需要卸载： 
kubectl delete -f manifests/setup   #直接删除了namespace
kubectl wait --for condition=Established --all CustomResourceDefinition --namespace=monitoring
kubectl delete -f manifests/

如果条件允许，在机房部署2-3个管理服务器，这样可以部署各种工具平台。而且可以管理训练网络，业务网络，数据网络，带外网络。

NFS部分：
1.NFS服务端：
apt install -y nfs-kernel-server
mkdir /nfs
chmod -R 777 /nfs # 设置文件权限。其实感觉755应该就可以了。
cat > /etc/exports <<EOF
/nfs   *(rw,sync,insecure,no_subtree_check,no_root_squash)
EOF

systemctl enable nfs-server.service
systemctl restart nfs-server.service

#在安装 NFS 服务器时，已包含常用的命令行工具，无需额外安装
#显示已经 mount 到本机 NFS 目录的客户端机器
showmount -e localhost
#将配置文件中的目录全部重新 export 一次，无需重启服务
exportfs -rv
#查看 NFS 的运行状态
nfsstat
#查看 rpc 执行信息，可以用于检测 rpc 运行情况
rpcinfo

2.NFS客户端：
apt install -y nfs-common
#显示指定的 NFS 服务器(假设 IP 地址为 172.18.6.69)上 export 出来的目录
showmount -e 172.18.6.69

# 每个节点创建共享存储文件夹：
mkdir /nfs;
mkdir -p /nfs/alertmanager /nfs/grafana /nfs/prometheus
chmod -R 777 /nfs

#假设 NFS 服务器 IP为 172.18.6.69，可以如下设置挂载  
sudo mount -t nfs 172.18.6.69:/nfs /nfs


配置持久化卷，添加3个文件：
kubectl apply -f kube-prometheus-storage-class.yaml
########################################
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: prometheus-storage
provisioner: kubernetes.io/no-provisioner
volumeBindingMode: WaitForFirstConsumer
########################################

kubectl apply -f kube-prometheus-pv.yaml
########################################
apiVersion: v1
kind: PersistentVolume
metadata:
  name: prometheus-pv
spec:
  capacity:
    storage: 40Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: prometheus-storage
  nfs:
    path: /nfs/prometheus
    server: 172.18.6.69    # 修改成正确的IP
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: alertmanager-pv
spec:
  capacity:
    storage: 2Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: prometheus-storage
  nfs:
    path: /nfs/alertmanager
    server: 172.18.6.69    # 修改成正确的IP
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: grafana-pv
spec:
  capacity:
    storage: 8Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: prometheus-storage
  nfs:
    path: /nfs/grafana
    server: 172.18.6.69    # 修改成正确的IP
########################################

kubectl apply -f kube-prometheus-values.yaml
########################################
apiVersion: v1
prometheus:
  prometheusSpec:
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: prometheus-storage
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 40Gi
alertmanager:
  alertmanagerSpec:
    storage:
      volumeClaimTemplate:
        spec:
          storageClassName: prometheus-storage
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 2Gi
grafana:
  persistence:
    enabled: true
    storageClassName: prometheus-storage
    accessModes: ["ReadWriteOnce"]
    size: 8Gi
########################################

# 需要修改的文件：
vim manifests/alertmanager-service.yaml
kubectl apply -f manifests/alertmanager-service.yaml 
########################################
apiVersion: v1
kind: Service
metadata:
  labels:
    app.kubernetes.io/component: alert-router
    app.kubernetes.io/instance: main
    app.kubernetes.io/name: alertmanager
    app.kubernetes.io/part-of: kube-prometheus
    app.kubernetes.io/version: 0.28.1
  name: alertmanager-main
  namespace: monitoring
spec:
  ports:
  - name: web
    nodePort: 30200    # 添加
    port: 9093
    targetPort: web
  - name: reloader-web
    port: 8080
    targetPort: reloader-web
  selector:
    app.kubernetes.io/component: alert-router
    app.kubernetes.io/instance: main
    app.kubernetes.io/name: alertmanager
    app.kubernetes.io/part-of: kube-prometheus
  sessionAffinity: ClientIP
  type: NodePort     # 添加
########################################

vim manifests/grafana-service.yaml
kubectl apply -f manifests/grafana-service.yaml
########################################
apiVersion: v1
kind: Service
metadata:
  labels:
    app.kubernetes.io/component: grafana
    app.kubernetes.io/name: grafana
    app.kubernetes.io/part-of: kube-prometheus
    app.kubernetes.io/version: 12.0.1
  name: grafana
  namespace: monitoring
spec:
  ports:
  - name: http
    nodePort: 30300   # 添加
    port: 3000
    targetPort: http
  selector:
    app.kubernetes.io/component: grafana
    app.kubernetes.io/name: grafana
    app.kubernetes.io/part-of: kube-prometheus
  type: NodePort    # 添加
########################################


vim manifests/prometheus-service.yaml
kubectl apply -f manifests/prometheus-service.yaml
########################################
apiVersion: v1
kind: Service
metadata:
  labels:
    app.kubernetes.io/component: prometheus
    app.kubernetes.io/instance: k8s
    app.kubernetes.io/name: prometheus
    app.kubernetes.io/part-of: kube-prometheus
    app.kubernetes.io/version: 3.4.0
  name: prometheus-k8s
  namespace: monitoring
spec:
  ports:
  - name: web
    nodePort: 31922      # 添加
    port: 9090
    targetPort: web
  - name: reloader-web
    port: 8080
    targetPort: reloader-web
  selector:
    app.kubernetes.io/component: prometheus
    app.kubernetes.io/instance: k8s
    app.kubernetes.io/name: prometheus
    app.kubernetes.io/part-of: kube-prometheus
  sessionAffinity: ClientIP
  type: NodePort        # 添加
########################################

kubectl -n monitoring get service
NAME                    TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)                         AGE
alertmanager-main       NodePort    10.233.48.95    <none>        9093:30200/TCP,8080:30574/TCP   104m
alertmanager-operated   ClusterIP   None            <none>        9093/TCP,9094/TCP,9094/UDP      97m
blackbox-exporter       ClusterIP   10.233.28.168   <none>        9115/TCP,19115/TCP              104m
grafana                 NodePort    10.233.34.41    <none>        3000:30300/TCP                  104m
kube-state-metrics      ClusterIP   None            <none>        8443/TCP,9443/TCP               104m
node-exporter           ClusterIP   None            <none>        9100/TCP                        104m
prometheus-adapter      ClusterIP   10.233.6.253    <none>        443/TCP                         104m
prometheus-k8s          NodePort    10.233.62.225   <none>        9090:31922/TCP,8080:32293/TCP   104m
prometheus-operated     ClusterIP   None            <none>        9090/TCP                        97m
prometheus-operator     ClusterIP   None            <none>        8443/TCP                        104m


# 直接在浏览器，打开grafana：
172.18.6.70：30300/
admin
admin


增加其它CPU节点的监控，增加GPU卡的监控，增加模型训练推理的监控。
增加数据库，对象存储的监控。

```
