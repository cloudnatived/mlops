



1. kubespray, kubespray-2.29.0, ingress-nginx, rancher
2. Kubernetes Metrics Server, 
3. kube-prometheus, 
4. krew, 
5. istio, 
6. kubeflow, kubeflow manifests-1.10.2
7. kuberay




```


1. kubespray
1.1 kubespray
kubespray-2.29.0

# 参考资料：
Kubespray部署 k8s v1.24.x集群    https://www.cnblogs.com/ggborn-001/p/18985663
kubespray离线k8s部署方案        https://www.cnblogs.com/ggborn-001/p/18989590

https://github.com/kubernetes-sigs/kubespray/releases/tag/v2.29.0                # 安装文档地址
https://github.com/kubernetes-sigs/kubespray/archive/refs/tags/v2.29.0.tar.gz    # 配置文件下载地址

# 相关组件的版本：
    kubernetes 1.33.5
    etcd 3.5.22
    docker 28.3
    containerd 2.1.4
    cri-o 1.33.4
    cni-plugins 1.8.0
    calico 3.30.3
    cilium 1.18.2
    flannel 0.27.3
    kube-ovn 1.12.21
    kube-router 2.1.1
    multus 4.2.2
    kube-vip 0.8.0
    cert-manager 1.15.3
    coredns 1.12.0
    ingress-nginx 1.13.3
    argocd 2.14.5
    helm 3.18.4
    metallb 0.13.9
    registry 2.8.1
    aws-ebs-csi-plugin 0.5.0
    azure-csi-plugin 1.10.0
    cinder-csi-plugin 1.30.0
    gcp-pd-csi-plugin 1.9.2
    local-path-provisioner 0.0.32
    local-volume-provisioner 2.5.0
    node-feature-discovery 0.16.4


# 安装完操作系统后，配置IP。
ip link set enp2s0f0 up
ip addr add 172.18.6.70/24 dev enp2s0f0
ip route add default via 172.18.6.1

# 写入网卡配置文件。
cat > /etc/netplan/50-cloud-init.yaml <<EOF
# network: {config: disabled}
network:
  ethernets:
    #enp2s0f0:
    eno1:
      dhcp4: false
      addresses:
        #- 172.18.6.69/24
        - 10.0.10.147/24
      routes:
        - to: default
          #via: 172.18.6.1
          via: 10.0.10.1
      nameservers:
         addresses: [8.8.8.8, 168.95.1.1]
  version: 2
EOF

# 使用netplan应用网络配置
netplan apply

# 使用清华大学的ubuntu-24.04.3的源
cat > /etc/apt/sources.list.d/ubuntu.sources <<EOF
Types: deb
URIs: https://mirrors.tuna.tsinghua.edu.cn/ubuntu/
Suites: noble noble-updates noble-security
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF


# 目前内核版本号
root@x:~# uname -a
Linux x 6.8.0-71-generic #71-Ubuntu SMP PREEMPT_DYNAMIC Tue Jul 22 16:52:38 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux

# 升级之后将变成的版本号。
root@x:~# uname -a
Linux x 6.8.0-78-generic #78-Ubuntu SMP PREEMPT_DYNAMIC Tue Aug 12 11:34:18 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux

# ubuntu-24.04.3默认操作系统版本信息。
root@x:~# cat /etc/*releas*
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=24.04
DISTRIB_CODENAME=noble
DISTRIB_DESCRIPTION="Ubuntu 24.04.3 LTS"
PRETTY_NAME="Ubuntu 24.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.3 LTS (Noble Numbat)"
VERSION_CODENAME=noble
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=noble
LOGO=ubuntu-logo

# 设置基础配置：
sed -i 's/SELINUX=enforcing/SELINUX=disabled/g' /etc/selinux/config;
sed -i 's/GSSAPIAuthentication yes/GSSAPIAuthentication no/g' /etc/ssh/sshd_config;
echo "GSSAPIAuthentication no" >> /etc/ssh/sshd_config;
sed -i 's/UseDNS yes/UseDNS no/g' /etc/ssh/sshd_config;
cat >> /etc/ssh/sshd_config <<EOF
UseDNS no
PermitRootLogin yes
EOF

systemctl restart ssh;

cat >> /etc/profile <<EOF
ulimit -S -c 0 > /dev/null 2>&1
ulimit -n 10240
ulimit -u 77823
EOF

cat >> /etc/sysctl.conf <<EOF
net.ipv4.ip_forward=1
net.ipv4.conf.all.rp_filter=0
net.ipv4.conf.default.rp_filter=0
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
net.ipv6.conf.lo.disable_ipv6 = 1
net.ipv4.conf.all.forwarding = 1
net.ipv6.conf.all.forwarding = 1

net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1

vm.swappiness=0
vm.overcommit_memory=1
fs.inotify.max_user_watches=524288
fs.inotify.max_user_instances=8192
EOF

创建 Kubernetes sysctl 配置
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.ipv4.ip_forward = 1
net.ipv4.conf.all.rp_filter = 0
net.ipv4.conf.default.rp_filter = 0
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
net.ipv6.conf.lo.disable_ipv6 = 1
net.ipv4.conf.all.forwarding = 1
net.ipv6.conf.all.forwarding = 1
vm.swappiness = 0
vm.overcommit_memory = 1
fs.inotify.max_user_watches = 524288
fs.inotify.max_user_instances = 8192
kernel.keys.root_maxbytes = 25000000
kernel.keys.root_maxkeys = 1000000
kernel.panic = 10
kernel.panic_on_oops = 1
vm.panic_on_oom = 0
net.ipv4.ip_local_reserved_ports = 30000-32767
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
EOF

ssh-keygen -t rsa -N "";
cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys;

# 写入固定配置的resolv配置文件
rm -rf /etc/resolv.conf

cat > /etc/resolv.conf <<EOF
nameserver 8.8.8.8
nameserver 168.95.1.1
EOF

# 更新源
# 可能会更新内核。注意，安装完nvidia的GPU驱动之后，不要升级内核，否则需要重新安装nvidia的GPU驱动。
apt update -y;
apt list --upgradable;
apt upgrade -y;

# 运行在init 3
systemctl isolate multi-user.target;
systemctl isolate runlevel3.target;
ln -sf /lib/systemd/system/multi-user.target /etc/systemd/system/default.target;
systemctl set-default multi-user.target;

#关闭不需要的服务：
systemctl list-unit-files |awk '{ print $1,$2 }'|grep enable|egrep -v "ssh|multi|systemd-resolved|wpa_" |awk '{ print $1}'|xargs -i systemctl disable {};

#确认服务已关闭：
systemctl list-unit-files |awk '{print $1,$2}'|grep enabled;

uname -r         # 查看内核版本
lsb_release -a   # 查看发行版信息（Ubuntu/Debian/CentOS等）

# 确认安装了gcc make
apt install -y gcc make g++ net-tools

apt install -y python3-pip python3 python3-netaddr wget git;
apt install -y python3-dev;
pip install --upgrade pip;

# 解决ubuntu24.04 使用pip时的信息提醒：error: externally-managed-environment
mv /usr/lib/python3.12/EXTERNALLY-MANAGED /usr/lib/python3.12/EXTERNALLY-MANAGED.bk

# 设置清华大学的apt源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple;
mkdir -p /root/.config/pip;
cat > /root/.config/pip/pip.conf <<EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
EOF

cat >> /etc/hosts <<EOF
10.0.10.148 xinhua148
10.0.10.153 xinhua153
10.0.10.155 xinhua155
10.0.10.173 xinhua173
10.0.10.210 xinhua210
EOF


# 重启一次。
reboot

# 时间同步
apt-get install -y ntpdate
ntpdate cn.pool.ntp.org ntp1.aliyun.com time1.apple.com

# 配置 rc-local，来执行开机启动:
vim /etc/systemd/system/rc-local.service  # 创建这个文件

touch /etc/systemd/system/rc-local.service
touch /etc/rc.local
chmod 775 /etc/rc.local
cat >> /etc/systemd/system/rc-local.service << EOF
[Unit]
Description=/etc/rc.local Compatibility
ConditionPathExists=/etc/rc.local
[Service]
Type=forking
ExecStart=/etc/rc.local start
TimeoutSec=0
StandardOutput=tty
RemainAfterExit=yes
#SysVStartPriority=99
[Install]
WantedBy=multi-user.target
EOF

touch /etc/rc.local
chmod 0775 /etc/rc.local

# 如果如要写入到 /etc/rc.local 文件:
cat >> /etc/rc.local << EOF
#!/bin/bash
nv-hostengine --service-account nvidia-dcgm -b ALL
EOF

systemctl enable rc-local.service
systemctl restart rc-local.service

cd /opt;
wget https://github.com/kubernetes-sigs/kubespray/archive/refs/heads/release-2.29.zip;   # 本次安装的版本
wget https://github.com/kubernetes-sigs/kubespray/archive/refs/tags/v2.23.3.tar.gz;      # 历史版本
wget https://github.com/kubernetes-sigs/kubespray/archive/refs/tags/v2.29.0.tar.gz;      # 本次安装的版本

#安装 kubespray 依赖
cd /opt/kubespray-2.29.0;

# 安装依赖
pip install -r requirements.txt;

# 本次安装，这两个命令都不需要运行：
#apt install ansible-core
#pip install ansible-core==2.17.3

pip install -r requirements.txt

# 出现报错：ERROR: Cannot uninstall cryptography 41.0.7, RECORD file not found. Hint: The package was installed by debian.
# 先用 apt 卸载，再用 pip 安装
apt remove python3-cryptography

# 忽略已安装版本（不推荐）
pip install --ignore-installed -r requirements.txt

# 以下这部分问题，不一定会出现。如果在系统基础安装包，未安装好的情况下，会出现：
----------------------------------------------------
# 安装 ansible.posix 集合
ansible-galaxy collection install ansible.posix

# 安装其他Kubespray可能需要的集合
ansible-galaxy collection install community.general
ansible-galaxy collection install kubernetes.core

# 安装 ansible.utils 集合
ansible-galaxy collection install ansible.utils

# 同时安装其他可能缺少的集合
ansible-galaxy collection install ansible.posix community.general kubernetes.core
----------------------------------------------------


# 复制一份 自己的配置
cd /opt/kubespray-2.29.0;
cp -au /opt/kubespray-2.29.0/inventory/sample /opt/kubespray-2.29.0/inventory/bbc;

# 修改配置 hosts.yaml 是这样部署了。
vim /opt/kubespray-2.29.0/inventory/bbc/hosts.yaml 
all:
  hosts:
    node1:
      ansible_host: anhua69
      ip: 172.18.6.69
      access_ip: 172.18.6.69
    node2:
      ansible_host: anhua70
      ip: 172.18.6.70
      access_ip: 172.18.6.70
    node3:
      ansible_host: anhua71
      ip: 172.18.6.71
      access_ip: 172.18.6.71     
  children:
    kube_control_plane:
      hosts:
        node1:
        node2:
        node3:
    kube_node:
      hosts:
        node1:
        node2:
        node3:
    etcd:
      hosts:
        node1:
        node2:
        node3:
    k8s_cluster:
      children:
        kube_control_plane:
        kube_node:
        calico_rr:
    calico_rr:
      hosts: {}



vim /opt/kubespray-2.23.3/inventory/bbc/group_vars/all/all.yml; #历史版本
vim /opt/kubespray-2.24.1/inventory/bbc/group_vars/all/all.yml; #历史版本
vim /opt/kubespray-2.29.0/inventory/bbc/group_vars/all/all.yml;
++++++++++++++++++++++++++++++++++++++++
# 打开下面这个选项:
loadbalancer_apiserver_localhost: true
++++++++++++++++++++++++++++++++++++++++

# 修改如下配置，海外的云服务器不要设置这两个镜像，海外的云服务器不要设置这两个镜像。: 
vim /opt/kubespray-2.23.3/inventory/bbc/group_vars/all/docker.yml; #历史版本
vim /opt/kubespray-2.24.1/inventory/bbc/group_vars/all/docker.yml; #历史版本
vim /opt/kubespray-2.29.0/inventory/bbc/group_vars/all/docker.yml;

++++++++++++++++++++++++++++++++++++++++
docker_registry_mirrors:
  - https://registry.docker-cn.com
  - https://mirror.aliyuncs.com
++++++++++++++++++++++++++++++++++++++++

# 以下两段都需要注释掉：
vim /opt/kubespray-2.23.3/extra_playbooks/roles/kubernetes/preinstall/tasks/0040-verify-settings.yml; #历史版本
vim /opt/kubespray-2.24.1/extra_playbooks/roles/kubernetes/preinstall/tasks/0040-verify-settings.yml; #历史版本
vim /opt/kubespray-2.29.0/extra_playbooks/roles/kubernetes/preinstall/tasks/0040-verify-settings.yml;
++++++++++++++++++++++++++++++++++++++++
- name: Stop if either kube_control_plane or kube_node group is empty
  assert:
    that: "groups.get('{{ item }}')"
  with_items:
    - kube_control_plane
    - kube_node
  run_once: true
  when: not ignore_assert_errors

- name: Stop if etcd group is empty in external etcd mode
  assert:
    that: groups.get('etcd')
    fail_msg: "Group 'etcd' cannot be empty in external etcd mode"
  run_once: true
  when:
    - not ignore_assert_errors
    - etcd_deployment_type != "kubeadm"
++++++++++++++++++++++++++++++++++++++++  # 如果是云服务器，应该关闭内存检查。把以下内容全部注释掉，使其不生效。
- name: Stop if memory is too small for masters
  assert:
    that: ansible_memtotal_mb >= minimal_master_memory_mb
  ignore_errors: "{{ ignore_assert_errors }}"
  when: inventory_hostname in groups['kube-master']

- name: Stop if memory is too small for nodes
  assert:
    that: ansible_memtotal_mb >= minimal_node_memory_mb
  ignore_errors: "{{ ignore_assert_errors }}"
  when: inventory_hostname in groups['kube-node']
++++++++++++++++++++++++++++++++++++++++

vim /opt/kubespray-2.29.0/inventory/bbc/group_vars/all/mirror.yml

cat > /opt/kubespray-2.29.0/inventory/bbc/group_vars/all/mirror.yml <<EOF
gcr_image_repo: "gcr.m.daocloud.io"
kube_image_repo: "k8s.m.daocloud.io"
docker_image_repo: "docker.m.daocloud.io"
quay_image_repo: "quay.m.daocloud.io"
github_image_repo: "ghcr.m.daocloud.io"
files_repo: "https://files.m.daocloud.io"
EOF

cat  > /opt/kubespray-2.29.0/inventory/bbc/group_vars/etcd.yml <<EOF
etcd_deployment_type: host
EOF

vim /opt/kubespray-2.29.0/inventory/bbc/group_vars/k8s_cluster/k8s-cluster.yml
auto_renew_certificates: true

# 本次安装，没有使用代理。如果需要设置代理：
vim /opt/kubespray-2.23.3/extra_playbooks/inventory/bbc/group_vars/all/all.yml; #历史版本
vim /opt/kubespray-2.24.1/extra_playbooks/inventory/bbc/group_vars/all/all.yml; #历史版本
http_proxy: "http://192.168.1.5:7890"
https_proxy: "http://192.168.1.5:7890"
no_proxy: "http://localhost:8080/,192.168.*.*,*.local,*.localhost*,localhost,127.0.0.1,192.168.1.100"

# 本次安装，没有使用代理。如果需要设置代理：
--------------------------
export http_proxy=http://192.168.1.5:7890;
export https_proxy=https://192.168.1.5:7890;
----------------------------



# 在所有节点上执行
modprobe br_netfilter

# 验证模块是否加载
lsmod | grep br_netfilter

# 确保模块在启动时自动加载
echo 'br_netfilter' | sudo tee /etc/modules-load.d/k8s.conf

# 1. 加载必要的内核模块
modprobe br_netfilter
modprobe overlay

# 2. 验证模块加载
echo "已加载的内核模块:"
lsmod | grep -E '(br_netfilter|overlay)'



# 在主节点生成密钥
ssh-keygen -t rsa

# 复制公钥到所有节点（包括自己）
ssh-copy-id -o StrictHostKeyChecking=no root@172.18.6.69
ssh-copy-id -o StrictHostKeyChecking=no root@172.18.6.70
ssh-copy-id -o StrictHostKeyChecking=no root@172.18.6.71 

ip="10.0.10.153 10.0.10.148 10.0.10.155 10.0.10.173 10.0.10.210"; for i in $ip ; do ssh $i "hostname" ; done
ip="10.0.10.153 10.0.10.148 10.0.10.155 10.0.10.173 10.0.10.210"; for i in $ip ; do ssh $i "date" ; done
ip="10.0.10.153 10.0.10.148 10.0.10.155 10.0.10.173 10.0.10.210"; for i in $ip ; do ssh $i "apt install -y ntpdate; ntpdate cn.pool.ntp.org ntp1.aliyun.com time1.apple.com" ; done


# 需要按照后面的离线安装，下载文件和容器镜像，才能kubespray-2.29.0 顺利完成。 
cd /opt/kubespray-2.29.0;
# 使用cilium网络
ansible-playbook -i inventory/bbc/hosts.yaml --become --become-user=root -e kube_network_plugin=cilium cluster.yml

# 使用flannel网络
ansible-playbook -i inventory/bbc/hosts.yaml --become --become-user=root -e kube_network_plugin=flannel cluster.yml   # 本次安装使用的命令。

#重新安装，增加1个计算节点。
ansible-playbook -i inventory/bbc/hosts.yaml --become --become-user=root -e kube_network_plugin=flannel cluster.yml



# 部署过程中会遇到的问题:

参考资料:
Kubespray部署k8s v1.24.x集群    https://www.cnblogs.com/ggborn-001/p/18985663
kubespray离线k8s部署方案        https://www.cnblogs.com/ggborn-001/p/18989590

# 使用kubespray:v2.29.0 容器镜像:
docker pull quay.m.daocloud.io/kubespray/kubespray:v2.29.0      # 470MB

# 查看containerd系统服务:
journalctl -u containerd.service -f

# 离线安装，下载文件和容器镜像:
# 修改files.list文件，加上files.m.daocloud.io前缀
cd /opt/kubespray-2.29.0/contrib/offline;
bash generate_list.sh
在temp目录下，生成 files.list 和 images.list，这2个文件。

sed -i "s#https://#https://files.m.daocloud.io/#g" files.list
 
# 修改images.list文件，修改成daocloud的镜像加速配置
sed -i "s@quay.io@quay.m.daocloud.io@g" images.list
sed -i "s@docker.io@docker.m.daocloud.io@g" images.list
sed -i "s@registry.k8s.io@k8s.m.daocloud.io@g" images.list
sed -i "s@ghcr.io@ghcr.m.daocloud.io@g" images.list

# 执行以下命令将依赖的静态文件全部下载到 temp/files 目录下
wget -x -P temp/files -i temp/files.list

# 下载镜像:
# 会下载52个镜像:
cat images.list |xargs -i nerdctl pull {}
cat images.list |xargs -i docker pull {}

# 把文件拷贝到/tmp/releases/
mkdir /tmp/releases/
find ./ -type f|xargs -i cp {} /tmp/releases/

# 未测试:
nerdctl images | grep ghcr.m.daocloud.io | awk '{print $1":"$2}' | while read image; do
new_image=$(echo $image | sed 's#旧标签#新标签#g')
nerdctl tag $image $new_image
done

# 把下载到本机的docker镜像打包，已测试:
while read -r A B _; do
    #echo "第一列(A): $A"
    #echo "第二列(B): $B"
    #nerdctl save $A:$B -o `echo $A |awk -F '/' '{ print $NF}'`..$B
    docker save $A:$B -o `echo $A |awk -F '/' '{ print $NF}'`..$B
    # 处理完第一行后可以加 break 终止循环
    #break
#done < nerdctl images |grep -v -E 'none|PLATFORM'
done < docker images |grep -v -E 'none|PLATFORM'

# 把当面目录下的容器镜像导入到主机:
ls |xargs -i nerdctl load -i {}


cat >> /etc/profile <<EOF
source <(kubectl completion bash)
source <(nerdctl completion bash)
EOF

可能用到的命令行
重新生成管理员令牌：
# 重新生成 admin.conf
sudo kubeadm init phase kubeconfig admin --config /etc/kubernetes/kubeadm-config.yaml

# 或者重新生成整个配置
sudo kubeadm init phase kubeconfig all --config /etc/kubernetes/kubeadm-config.yaml

# 如果集群损坏严重:
# 1. 清理现有 Kubernetes 集群（保留 OS，但移除所有 K8s 组件）
ansible-playbook -i inventory/bbc/hosts.yaml \
  --become --become-user=root \
  reset.yml

# 2. 重新部署集群（使用 flannel 网络插件）
ansible-playbook -i inventory/bbc/hosts.yaml \
  --become --become-user=root \
  -e kube_network_plugin=flannel \
  cluster.yml


1.2 ingress-nginx
ingress-nginx controller-v1.14.0
https://github.com/kubernetes/ingress-nginx/releases/tag/controller-v1.14.0
https://github.com/kubernetes/ingress-nginx/archive/refs/tags/controller-v1.14.0.tar.gz
https://kubernetes.github.io/ingress-nginx/deploy/

# 部署:
# kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.14.0/deploy/static/provider/cloud/deploy.yaml   # 官方命令
wget https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.14.0/deploy/static/provider/cloud/deploy.yaml                 # 实际操作步骤 
sed -i "s#image: #image: m.daocloud.io/#g" deploy.yaml
kubectl apply -f deploy.yaml

nerdctl pull m.daocloud.io/registry.k8s.io/ingress-nginx/controller:v1.14.0@sha256:e4127065d0317bd11dc64c4dd38dcf7fb1c3d72e468110b4086e636dbaac943d
nerdctl save m.daocloud.io/registry.k8s.io/ingress-nginx/kube-webhook-certgen:v1.6.4@sha256:bcfc926ed57831edf102d62c5c0e259572591df4796ef1420b87f9cf6092497f -o controller..v1.14.0@sha256:e4127065d0317bd11dc64c4dd38dcf7fb1c3d72e468110b4086e636dbaac943d
nerdctl save m.daocloud.io/registry.k8s.io/ingress-nginx/controller:v1.14.0@sha256:e4127065d0317bd11dc64c4dd38dcf7fb1c3d72e468110b4086e636dbaac943d -o controller..v1.14.0@sha256:e4127065d0317bd11dc64c4dd38dcf7fb1c3d72e468110b4086e636dbaac943d
nerdctl pull m.daocloud.io/registry.k8s.io/ingress-nginx/kube-webhook-certgen:v1.6.4@sha256:bcfc926ed57831edf102d62c5c0e259572591df4796ef1420b87f9cf6092497f
nerdctl save m.daocloud.io/registry.k8s.io/ingress-nginx/kube-webhook-certgen:v1.6.4@sha256:bcfc926ed57831edf102d62c5c0e259572591df4796ef1420b87f9cf6092497f -o kube-webhook-certgen..v1.6.4@sha256:bcfc926ed57831edf102d62c5c0e259572591df4796ef1420b87f9cf6092497f

# 验证:
kubectl get pods -n ingress-nginx
kubectl get svc -n ingress-nginx

使用 kubectl patch 修改为 NodePort
kubectl patch svc ingress-nginx-controller -n ingress-nginx -p '{"spec": {"type": "NodePort"}}'
kubectl get svc -n ingress-nginx ingress-nginx-controller
curl -H "Host: your-host.com" http://<NODE_IP>:31234      # 示范
curl -H "Host: your-host.comm" http://10.0.10.153:32415  
telnet 10.0.10.153 32415

kubectl patch svc ingress-nginx-controller -n ingress-nginx -p '{"spec": {"type": "LoadBalancer"}}'     # 恢复为 LoadBalancer（如需）



1.3 rancher

docker pull registry.cn-hangzhou.aliyuncs.com/rancher/rancher:v2.8.4
docker pull rancher/rancher:v2.13-dc792f277d13955ed1de3081adc617af0bd28f2c-head
docker pull hub.rat.dev/rancher/rancher:v2.13-dc792f277d13955ed1de3081adc617af0bd28f2c-head    # 可执行

docker run -d --restart=unless-stopped \
  -p 80:80 -p 443:443 \
  --privileged \
  -e CATTLE_SYSTEM_DEFAULT_REGISTRY=registry.cn-hangzhou.aliyuncs.com \
  --name rancher \
  registry.cn-hangzhou.aliyuncs.com/rancher/rancher:v2.8.4

docker run -d --restart=unless-stopped \
  -p 80:80 -p 443:443 \
  --privileged \
  --name rancher \
  hub.rat.dev/rancher/rancher:v2.13-dc792f277d13955ed1de3081adc617af0bd28f2c-head

https://ranchermanager.docs.rancher.com/zh/getting-started/quick-start-guides/deploy-workloads/workload-ingress







2. Kubernetes Metrics Server
Kubernetes Metrics Server v0.8.0
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
        image: m.daocloud.io/registry.k8s.io/metrics-server/metrics-server:v0.8.0    # registry.k8s.io/metrics-server/metrics-server:v0.8.0
################################
registry.k8s.io/metrics-server/metrics-server:v0.8.0
m.daocloud.io/registry.k8s.io/metrics-server/metrics-server:v0.8.0


2.1 基础状态检查命令
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

2.2 资源指标查询命令
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

2.3 诊断和故障排除命令
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

2.4 高级测试命令
直接访问Metrics API
# 获取所有节点的metrics数据
kubectl get --raw /apis/metrics.k8s.io/v1beta1/nodes | jq .

# 获取所有pods的metrics数据
kubectl get --raw /apis/metrics.k8s.io/v1beta1/pods | jq .

# 获取特定命名空间的pods metrics
kubectl get --raw /apis/metrics.k8s.io/v1beta1/namespaces/kube-system/pods | jq .

# 检查Pod安全配置
kubectl get pod -n kube-system -l k8s-app=metrics-server -o yaml | grep -A5 securityContext

# 查看node的资源状况：
kubectl top node
NAME    CPU(cores)   CPU(%)   MEMORY(bytes)   MEMORY(%)   
node1   604m         8%       2557Mi          4%          
node2   512m         6%       2430Mi          3%          
node3   493m         6%       2334Mi          3%         

Metrics Server与其他工具的对比

以下表格展示了Metrics Server与其他监控工具的对比：

| 工具           | 功能               | 数据存储 | 可视化支持 |
| -------------- | ------------------ | -------- | ---------- |
| Metrics Server | 集群资源度量API    | 不支持   | 不支持     |
| Prometheus     | 完整监控和报警系统 | 支持     | 支持       |
| Grafana        | 数据可视化工具     | 不支持   | 支持       |







3. kube-prometheus
3.1 kube-prometheus
kube-prometheus-0.15.0
kube-prometheus-0.16.0

主要参考文档：
https://www.cnblogs.com/niuben/p/18888238   主要是看的这个文档里的NFS。
https://cloud.tencent.com/developer/article/1780158


项目来自：
https://github.com/prometheus-operator/kube-prometheus

wget https://github.com/prometheus-operator/kube-prometheus/archive/refs/tags/v0.16.0.tar.gz

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
cd /opt/kube-prometheus-0.16.0/manifests;
sed -i "s#image: #image: m.daocloud.io/#g" *    # 大部分的使用 m.daocloud.io 代理进行下载。

# grafana 需要使用 hub.rat.dev 的镜像:
sed -i "s#m.daocloud.io/grafana/grafana:12.1.0#hub.rat.dev/grafana/grafana:12.1.0#g" grafana-deployment.yaml
kubectl apply -f grafana-deployment.yaml

按照官网，执行部署的命令：
# 步骤1
kubectl apply --server-side -f manifests/setup

# 步骤2
kubectl wait \
	--for condition=Established \
	--all CustomResourceDefinition \
	--namespace=monitoring

# 步骤3
kubectl apply -f manifests/

# kubectl wait --for condition=Established --all CustomResourceDefinition --namespace=monitoring

# 如果需要卸载： 
kubectl delete -f manifests/setup   #直接删除了namespace
kubectl wait --for condition=Established --all CustomResourceDefinition --namespace=monitoring
kubectl delete -f manifests/

如果条件允许，在机房部署2-3个管理服务器，这样可以部署各种工具平台。而且可以管理训练网络，业务网络，数据网络，带外网络。

3.2 NFS部分：
3.2.1 NFS服务端：
apt install -y nfs-kernel-server
# 每个节点创建共享存储文件夹：
mkdir /nfs;
mkdir -p /nfs/alertmanager /nfs/grafana /nfs/prometheus
chmod -R 0755 /nfs
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

3.2.2 NFS客户端：
apt install -y nfs-common
#显示指定的 NFS 服务器(假设 IP 地址为 172.18.6.69)上 export 出来的目录
showmount -e 172.18.6.69

# 每个节点创建共享存储文件夹：
mkdir /nfs;
mkdir -p /nfs/alertmanager /nfs/grafana /nfs/prometheus
chmod -R 0755 /nfs

#假设 NFS 服务器 IP为 172.18.6.69，可以如下设置挂载  
mount -t nfs 172.18.6.69:/nfs /nfs
mount -t nfs 10.0.10.153:/nfs /nfs

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







4. Krew 
# 中文文档：
https://krew.kubernetes.ac.cn/docs/user-guide/setup/install/

https://github.com/kubernetes-sigs/krew/releases/latest/download/krew-linux_amd64.tar.gz

macOS/Linux
Bash 或 ZSH shell
# 确保已安装 git。
# 运行此命令下载并安装 krew
(
  set -x; cd "$(mktemp -d)" &&
  OS="$(uname | tr '[:upper:]' '[:lower:]')" &&
  ARCH="$(uname -m | sed -e 's/x86_64/amd64/' -e 's/\(arm\)\(64\)\?.*/\1\2/' -e 's/aarch64$/arm64/')" &&
  KREW="krew-${OS}_${ARCH}" &&
  curl -fsSLO "https://github.com/kubernetes-sigs/krew/releases/latest/download/${KREW}.tar.gz" &&
  tar zxvf "${KREW}.tar.gz" &&
  ./"${KREW}" install krew
)

# 将 $HOME/.krew/bin 目录添加到你的 PATH 环境变量。为此，更新你的 .bashrc 或 .zshrc 文件并追加以下行
export PATH="${KREW_ROOT:-$HOME/.krew}/bin:$PATH"

运行 kubectl krew 检查安装。

krew使用
1.插件索引更新
kubectl krew update

2.插件搜索
kubectl krew search
kubectl krew search crt

3.安装插件
kubectl krew install get-all
kubectl krew install ns tail

4.查看已装插件
kubectl krew list

5.查看插件详情
kubectl krew info ns

6.插件更新
krew upgrade ns

7.使用插件--ns
kubectl ns weave
kubectl-ns default

8.使用插件--get-all
kubectl-get_all

9.使用插件--tail
kubectl-tail
kubectl-tail --ns default 
kubectl-tail --rs kubeapps-8fd98f6f5
kubectl-tail --rs kubeapps/kubeapps-8fd98f6f5 

查看安装目录
kubectl krew version

krew卸载
rm -rf ~/.krew








5. istio
istio-1.27.3
https://istio.io/latest/zh/docs/
https://istio.io/latest/zh/docs/setup/additional-setup/download-istio-release/

curl -L https://istio.io/downloadIstio | ISTIO_VERSION=1.27.3 TARGET_ARCH=x86_64 sh -

wget -c https://github.com/istio/istio/releases/download/1.27.3/istio-1.27.3-linux-amd64.tar.gz

# 安装文档地址：
https://istio.io/latest/zh/docs/setup/install/istioctl/   # 这里的说明是，使用 Istioctl 安装

最简单的选择是用下面命令安装 Istio 默认配置档：
$ istioctl install

此命令在 Kubernetes 集群上安装 default 配置档。 
default 配置档是建立生产环境的一个良好起点， 这和较大的 demo 配置档不同，后者常用于评估一组广泛的 Istio 特性。
可以配置各种设置来修改安装。比如，要启动访问日志：
$ istioctl install --set meshConfig.accessLogFile=/dev/stdout

其他的 Istio 配置档，可以通过在命令行传递配置档名称的方式，安装到集群。 例如，下面命令可以用来安装 demo 配置档。
$ istioctl install --set profile=demo

在安装 Istio 之前，可以用 manifest generate 子命令生成清单文件。
例如，使用以下命令为可以使用 kubectl 安装的 default 配置文件生成清单：
$ istioctl manifest generate > $HOME/generated-manifest.yaml

卸载 Istio
要从集群中完整卸载 Istio，运行下面命令：
$ istioctl uninstall --purge

或者，只移除指定的 Istio 控制平面，运行以下命令：
$ istioctl uninstall <your original installation options>

或
istioctl manifest generate <your original installation options> | kubectl delete --ignore-not-found=true -f -

控制平面的命名空间（例如：istio-system）默认不会被移除。 如果确认不再需要，用下面命令移除该命名空间：
$ kubectl delete namespace istio-system



# 如果镜像下载不了:
docker.io/istio/pilot:1.27.3
docker.io/istio/proxyv2:1.27.3
docker.io/istio/proxyv2:1.27.3
nerdctl pull m.daocloud.io/docker.io/istio/pilot:1.27.3
nerdctl pull m.daocloud.io/docker.io/istio/proxyv2:1.27.3
nerdctl tag m.daocloud.io/docker.io/istio/pilot:1.27.3 docker.io/istio/pilot:1.27.3
nerdctl tag m.daocloud.io/docker.io/istio/proxyv2:1.27.3 docker.io/istio/proxyv2:1.27.3


nerdctl save docker.io/istio/pilot:1.27.3 -o pilot..1.27.3
nerdctl save docker.io/istio/proxyv2:1.27.3 -o proxyv2..1.27.3

ip="10.0.10.148 10.0.10.155 10.0.10.173 10.0.10.210"; for i in $ip ; do scp pilot..1.27.3 $i:/Data/IMAGES/; ssh $i "nerdctl load -i /Data/IMAGES/pilot..1.27.3" ; done
ip="10.0.10.148 10.0.10.155 10.0.10.173 10.0.10.210"; for i in $ip ; do scp proxyv2..1.27.3 $i:/Data/IMAGES/; ssh $i "nerdctl load -i /Data/IMAGES/proxyv2..1.27.3" ; done







6. Kubeflow
Kubeflow manifests-1.10.2
kubeflow manifests-1.10.2
Kubeflow Platform 1.10.2
https://github.com/kubeflow/manifests/releases/tag/v1.10.2
https://github.com/kubeflow/manifests/archive/refs/tags/v1.10.2.tar.gz

/opt/manifests-1.10.2
grep -E "image: .+:" -R


# 先部署：
wget https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.26/deploy/local-path-storage.yaml
# 修改 local-path-storage.yaml 里的镜像地址:
        image: m.daocloud.io/docker.io/rancher/local-path-provisioner:v0.0.26
        image: m.daocloud.io/docker.io/library/busybox:1.37.0
kubectl apply -f local-path-storage.yaml

vim common/oidc-client/oidc-authservice/base/pvc.yaml #1.9.1不需要，manifests-1.10.2 中没有这个路径。
-------------------------------------------------------------------------------------
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: authservice-pvc
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: local-path   #添加storageClassName名称
  resources:
    requests:
      storage: 10Gi
-------------------------------------------------------------------------------------


vim apps/kfp-tekton/upstream/v1/third-party/minio/base/minio-pvc.yaml #1.9.1需要。
vim applications/pipeline/upstream/third-party/minio/base/minio-pvc.yaml  # manifests-1.10.2 位置。
-------------------------------------------------------------------------------------
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: minio-pvc
  namespace: kubeflow
spec:
  accessModes:
  - ReadWriteOnce
  storageClassName: local-path #添加storageClassName名称
  resources:
    requests:
      storage: 20Gi
-------------------------------------------------------------------------------------
# 如果需要手动执行的话，需要指明 namespace
kubectl -n kubeflow apply -f applications/pipeline/upstream/third-party/minio/base/minio-pvc.yaml


vim apps/katib/upstream/components/mysql/pvc.yaml #1.9.1需要。
vim applications/katib/upstream/components/mysql/pvc.yaml # manifests-1.10.2 位置。
-------------------------------------------------------------------------------------
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: katib-mysql
  namespace: kubeflow
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: local-path #添加storageClassName名称
  resources:
    requests:
      storage: 10Gi
-------------------------------------------------------------------------------------
# 如果需要手动执行的话，需要指明 namespace
kubectl -n kubeflow apply -f applications/katib/upstream/components/mysql/pvc.yaml

vim apps/kfp-tekton/upstream/v1/third-party/mysql/base/mysql-pv-claim.yaml #1.9.1需要。
vim applications/pipeline/upstream/third-party/mysql/base/mysql-pv-claim.yaml # manifests-1.10.2 位置。
-------------------------------------------------------------------------------------
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mysql-pv-claim
  namespace: kubeflow
spec:
  accessModes:
  - ReadWriteOnce
  storageClassName: local-path #添加storageClassName名称
  resources:
    requests:
      storage: 20Gi
-------------------------------------------------------------------------------------
# 如果需要手动执行的话，需要指明 namespace
kubectl -n kubeflow apply -f applications/pipeline/upstream/third-party/mysql/base/mysql-pv-claim.yaml 


# 检查当前的 PVC 状态
kubectl get pvc -n kubeflow

# 检查可用的存储类
kubectl get storageclass

# 如果缺少存储类，需要先创建 (举例):
kubectl apply -f storage-class.yaml

# 检查节点存储状态
kubectl describe nodes | grep -A 10 -B 5 "Storage"

# 检查 PV 状态
kubectl get pv


# 最终，正常运行时的 pv 和 pvc 信息:
-------------------------------------------------------------------------------------
$ kubectl get pv,pvc -A
NAME                                                        CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS   CLAIM                     STORAGECLASS   VOLUMEATTRIBUTESCLASS   REASON   AGE
persistentvolume/pvc-3bfa0749-2e7b-4b22-8606-b83f6d465053   20Gi       RWO            Delete           Bound    kubeflow/minio-pvc        local-path     <unset>                          5m14s
persistentvolume/pvc-5f564f39-0f40-49d0-b382-b084ae2506f9   20Gi       RWO            Delete           Bound    kubeflow/mysql-pv-claim   local-path     <unset>                          2m58s
persistentvolume/pvc-b0a67d8a-e4ef-4fdd-ae70-fea922140240   10Gi       RWO            Delete           Bound    kubeflow/katib-mysql      local-path     <unset>                          2m27s

NAMESPACE   NAME                                   STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   VOLUMEATTRIBUTESCLASS   AGE
kubeflow    persistentvolumeclaim/katib-mysql      Bound    pvc-b0a67d8a-e4ef-4fdd-ae70-fea922140240   10Gi       RWO            local-path     <unset>                 2m37s
kubeflow    persistentvolumeclaim/minio-pvc        Bound    pvc-3bfa0749-2e7b-4b22-8606-b83f6d465053   20Gi       RWO            local-path     <unset>                 5m19s
kubeflow    persistentvolumeclaim/mysql-pv-claim   Bound    pvc-5f564f39-0f40-49d0-b382-b084ae2506f9   20Gi       RWO            local-path     <unset>                 3m3s
-------------------------------------------------------------------------------------





# 需要手动下载 kustomize 并放到/usr/bin/目录下:
https://github.com/kubernetes-sigs/kustomize/archive/refs/tags/kustomize/v5.4.3.tar.gz
https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2Fv5.4.3/kustomize_v5.4.3_linux_amd64.tar.gz


# 在镜像地址前添加代理地址:
cd /opt/manifests-1.10.2;
find ./ -type f |xargs -i sed -i "s#image: gcr.io#image: m.daocloud.io/gcr.io#g" {};
find ./ -type f |xargs -i sed -i "s#image: ghcr.io#image: m.daocloud.io/ghcr.io#g" {};
find ./ -type f |xargs -i sed -i "s#image: quay.io#image: m.daocloud.io/quay.io#g" {};
find ./ -type f |xargs -i sed -i "s#image: registry.k8s.io#image: m.daocloud.io/registry.k8s.io#g" {};
find ./ -type f |xargs -i sed -i "s#image: ghcr.io#image: m.daocloud.io/ghcr.io#g" {};

find ./ -type f |xargs -i sed -i "s#image: spark#image: hub.rat.dev/spark#g" {};
find ./ -type f |xargs -i sed -i "s#image: kserve#image: hub.rat.dev/kserve#g" {};
find ./ -type f |xargs -i sed -i "s#image: kubeflownotebookswg#image: hub.rat.dev/kubeflownotebookswg#g" {};
find ./ -type f |xargs -i sed -i "s#image: mysql#image:hub.rat.dev/mysql#g" {};
find ./ -type f |xargs -i sed -i "s#image: postgres#image:hub.rat.dev/postgres#g" {};
find ./ -type f |xargs -i sed -i "s#image: kindest#image:hub.rat.dev/kindest#g" {};
find ./ -type f |xargs -i sed -i "s#image: prom#image:hub.rat.dev/prom#g" {};


while ! kustomize build example | kubectl apply -f -; do echo "Retrying to apply resources"; sleep 10; done
while ! kustomize build example | kubectl apply --server-side --force-conflicts -f -; do echo "Retrying to apply resources"; sleep 20; done  # 本次部署使用的命令


# 可能用到的命令:
kubectl get pv,pvc -A



# 查看非 Running 状态的 pod:
kubectl get pod -A |grep -E -v 'Runn|NAME'
kubectl get pod -A | awk '!/Runn|NAME/ {print $1, $2}' | xargs -n2 sh -c 'kubectl describe pod -n "$1" "$2"' _|grep "pulling image"   # 查看还有那些镜像没有完成下载

# 删掉非 Running 状态的 pod:
kubectl get pod -A | grep -E -v 'Runn|NAME' | awk '{print $1, $2}' | while read namespace pod; do
    echo "Deleting pod $pod in namespace $namespace"
    kubectl delete pod -n "$namespace" "$pod"
done

kubectl get pod -A | awk '!/Runn|NAME/ {print $1, $2}' | xargs -n2 sh -c 'kubectl delete pod -n "$1" "$2"' _


# 有些镜像的版本号为 :dummy ，要修改成2.5.0
kubectl -n kubeflow describe deployments.apps ml-pipeline-api-server |grep -i image
kubectl -n kubeflow get deployment cache-server -o yaml | grep "name:"
kubectl -n kubeflow get deployment ml-pipeline-visualizationserver -o jsonpath='{.spec.template.spec.containers[*].name}'

# 查询命令:
kubectl -n kubeflow get deployment cache-server -o jsonpath='{.spec.template.spec.containers[*].name}'
kubectl -n kubeflow get deployment metadata-writer -o jsonpath='{.spec.template.spec.containers[*].name}'
kubectl -n kubeflow get deployment ml-pipeline -o jsonpath='{.spec.template.spec.containers[*].name}'
kubectl -n kubeflow get deployment ml-pipeline-persistenceagent -o jsonpath='{.spec.template.spec.containers[*].name}'
kubectl -n kubeflow get deployment ml-pipeline-scheduledworkflow -o jsonpath='{.spec.template.spec.containers[*].name}'
kubectl -n kubeflow get deployment ml-pipeline-ui -o jsonpath='{.spec.template.spec.containers[*].name}'
kubectl -n kubeflow get deployment ml-pipeline-viewer-crd -o jsonpath='{.spec.template.spec.containers[*].name}'
kubectl -n kubeflow get deployment ml-pipeline-visualizationserver -o jsonpath='{.spec.template.spec.containers[*].name}'
kubectl -n kubeflow get deployment metadata-envoy-deployment -o jsonpath='{.spec.template.spec.containers[*].name}'

# 执行替换的命令:
kubectl -n kubeflow set image deployment/cache-server server=m.daocloud.io/ghcr.io/kubeflow/kfp-cache-server:2.5.0
kubectl -n kubeflow set image deployment/metadata-writer main=m.daocloud.io/ghcr.io/kubeflow/kfp-metadata-writer:2.5.0
kubectl -n kubeflow set image deployment/ml-pipeline ml-pipeline-api-server=m.daocloud.io/ghcr.io/kubeflow/kfp-api-server:2.5.0
kubectl -n kubeflow set image deployment/ml-pipeline-persistenceagent ml-pipeline-persistenceagent=m.daocloud.io/ghcr.io/kubeflow/kfp-persistence-agent:2.5.0
kubectl -n kubeflow set image deployment/ml-pipeline-scheduledworkflow ml-pipeline-scheduledworkflow=m.daocloud.io/ghcr.io/kubeflow/kfp-scheduled-workflow-controller:2.5.0
kubectl -n kubeflow set image deployment/ml-pipeline-ui ml-pipeline-ui=m.daocloud.io/ghcr.io/kubeflow/kfp-frontend:2.5.0
kubectl -n kubeflow set image deployment/ml-pipeline-viewer-crd ml-pipeline-viewer-crd=m.daocloud.io/ghcr.io/kubeflow/kfp-viewer-crd-controller:2.5.0
kubectl -n kubeflow set image deployment/ml-pipeline-visualizationserver ml-pipeline-visualizationserver=m.daocloud.io/ghcr.io/kubeflow/kfp-visualization-server:2.5.0
kubectl -n kubeflow set image deployment/metadata-envoy-deployment container=m.daocloud.io/ghcr.io/kubeflow/kfp-metadata-envoy:2.5.0

m.daocloud.io/ghcr.io/kubeflow/kfp-metadata-envoy:dummy

# 拷贝容器镜像:
ip="10.0.10.148 10.0.10.155 10.0.10.173 10.0.10.210"; for i in $ip ; do scp proxyv2..1.27.3 $i:/Data/IMAGES/; ssh $i "nerdctl load -i /Data/IMAGES/proxyv2..1.27.3" ; done

# 删除pvc
kubectl delete pvc katib-mysql minio-pvc mysql-pv-claim -n kubeflow


# Port-Forward
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80   # 不执行这一条，执行下面一条，修改成 nodeport

# 查看所有 NodePort 类型的 Service
kubectl -n istio-system patch svc/istio-ingressgateway -p '{"spec":{"type":"NodePort"}}'
kubectl -n istio-system get svc

# 登录的官方文档说明：
After running the command, you can access the Kubeflow Central Dashboard by doing the following:
1. Open your browser and visit http://localhost:8080. You should see the Dex login screen.
2. Log in with the default user's credentials. The default email address is user@example.com, and the default password is 12341234. '

# 示当前配置的容器运行时端点信息
crictl config --list




Kubeflow manifests-1.10.2 检查:

1. 检查是否有运行错误的 pod 。
kubectl get pod -A|grep -v Runn

2. 检查 containerd 服务，是否有报错信息。
journalctl -u containerd.service -f







Kubeflow manifests-1.10.2 功能性问题:

1. 建立 Volumes 和创建 notebook 的时候会出错。
--------------------------------------------------------------------------------------------
[403] Could not find CSRF co okie XSRF-TOKEN in therequest.http://121.40.245.182:7116/volumes/api/namespaces/kubeflow-user-exampleCom/pvCs
......
[200] undefined http://121.40.245.182:7116/volumes/api/namespaces/pvcs
......
Could not find CSRF cookie XSRF-TOKEN
......
No default Storage Class is set. Can't create new Disks for the new Notebook. Please use an Existing Disk.   '   #  点击 New notebook 会出现这个提示:
......
[403] Could not find CSRF cookie XSRF-TOKEN in the request. http://121.40.245.182:7116/jupyter/api/namespaces/kubeflow-user-example-com/notebooks

[200] undefined http://121.40.245.182:7116/volumes/api/namespaces/pvcs
--------------------------------------------------------------------------------------------


# 解决方法:
kubectl get storageclass
kubectl patch storageclass local-path -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'


# 镜像无法下载:
filebrowser/filebrowser:v2.25.0
nerdctl pull m.daocloud.io/docker.io/filebrowser/filebrowser:v2.25.0
nerdctl tag m.daocloud.io/docker.io/filebrowser/filebrowser:v2.25.0 filebrowser/filebrowser:v2.25.0
nerdctl save filebrowser/filebrowser:v2.25.0 -o filebrowser..v2.25.0

nerdctl pull ghcr.io/kubeflow/kubeflow/notebook-servers/jupyter-scipy:v1.10.0
nerdctl save ghcr.io/kubeflow/kubeflow/notebook-servers/jupyter-scipy:v1.10.0 -o jupyter-scipy..v1.10.0


ip="10.0.10.148 10.0.10.155 10.0.10.173 10.0.10.210"; for i in $ip ; do scp filebrowser..v2.25.0 $i:/Data/IMAGES/; ssh $i "nerdctl load -i /Data/IMAGES/filebrowser..v2.25.0" ; done
ip="10.0.10.148 10.0.10.155 10.0.10.173 10.0.10.210"; for i in $ip ; do scp jupyter-scipy..v1.10.0 $i:/Data/IMAGES/; ssh $i "nerdctl load -i /Data/IMAGES/jupyter-scipy..v1.10.0" ; done


# 如果需要的话，删除并重新创建 Released 状态的 PV
kubectl delete pv jupyter-storage
kubectl delete pv katib-mysql

# 如果需要的话，或者重新绑定（如果数据不重要）
kubectl patch pv jupyter-storage -p '{"spec":{"claimRef": null}}'
kubectl patch pv katib-mysql -p '{"spec":{"claimRef": null}}'


创建 notebook 的时候会出错。
-------------------------------------------------------------------------
Pending: no persistent volumes available for this claim and no storage class is set

Unschedulable: 0/5 nodes are available: pod has unbound immediate PersistentVolumeClaims. preemption: 0/5 nodes are available: 5 Preemption is not helpful for scheduling.

No default Storage Class is set. Can't create new Disks for the new Notebook. Please use an Existing Disk.   '

-------------------------------------------------------------------------

# 找到是哪个 deployment 部署的这个 pod 。 搜索 kubeflow 这个 namespace 下的所有 pod 的日志，在复杂系统中，初步过滤问题，很重要:
kubectl -n kubeflow get pods -o name | xargs -I {} sh -c 'echo "=== {} ==="; kubectl -n kubeflow logs {} 2>/dev/null | grep -i "j222-0" || true'

nerdctl pull tensorflow/tensorflow:2.5.1
nerdctl pull m.daocloud.io/docker.io/tensorflow/tensorflow:2.5.1
nerdctl tag m.daocloud.io/docker.io/tensorflow/tensorflow:2.5.1 tensorflow/tensorflow:2.5.1
nerdctl save tensorflow/tensorflow:2.5.1 -o tensorflow..2.5.1
ip="10.0.10.148 10.0.10.155 10.0.10.173 10.0.10.210"; for i in $ip ; do scp tensorflow..2.5.1 $i:/Data/IMAGES/; ssh $i "nerdctl load -i /Data/IMAGES/tensorflow..2.5.1" ; done

nerdctl pull m.daocloud.io/docker.io/rancher/local-path-provisioner:v0.0.26
nerdctl tag m.daocloud.io/docker.io/rancher/local-path-provisioner:v0.0.26 rancher/local-path-provisioner:v0.0.26
nerdctl save rancher/local-path-provisioner:v0.0.26 -o local-path-provisioner..v0.0.26
ip="10.0.10.148 10.0.10.155 10.0.10.173 10.0.10.210"; for i in $ip ; do scp local-path-provisioner..v0.0.26 $i:/Data/IMAGES/; ssh $i "nerdctl load -i /Data/IMAGES/local-path-provisioner..v0.0.26" ; done

docker pull busybox:1.37.0




2. 创建 Katib Experiments 报错。
[500] admission webhook "validator.experiment.katib.kubeflow.org" denied the request: spec.trialTemplate.trialParameters must be specified
......

#查看katib-ui日志，发现以下报错。
kubectl -n kubeflow logs -f katib-ui-949786bfd-zsg2t
-------------------------------------------------------------------------
2025/11/01 16:22:58 Serving the frontend dir /app/build
2025/11/01 16:22:58 Serving at 0.0.0.0:8080
2025/11/02 18:00:33 Sending file /app/build/static/index.html for url: /katib/
2025/11/02 18:01:23 Sending file /app/build/static/index.html for url: /katib/
2025/11/05 03:09:24 Sending file /app/build/static/index.html for url: /katib/
2025/11/05 03:09:39 CreateRuntimeObject from parameters failed: admission webhook "validator.experiment.katib.kubeflow.org" denied the request: spec.trialTemplate.trialParameters: Required value: must be specified



2025/11/05 03:10:26 CreateRuntimeObject from parameters failed: admission webhook "validator.experiment.katib.kubeflow.org" denied the request: spec.trialTemplate.trialParameters: Required value: must be specified
2025/11/05 03:12:28 CreateRuntimeObject from parameters failed: admission webhook "validator.experiment.katib.kubeflow.org" denied the request: [spec.trialTemplate.trialParameters[0]: Invalid value: "": name and reference must be specified and name must not contain '{' or '}', spec.trialTemplate.trialParameters[1]: Invalid value: "": name and reference must be specified and name must not contain '{' or '}', spec.trialTemplate: Invalid value: "": parameters: [${trialParameters.learningRate} ${trialParameters.momentum}] in spec.trialTemplate not found in spec.trialParameters: [{learningRate  } {momentum  }]]
-------------------------------------------------------------------------

nerdctl pull m.daocloud.io/docker.io/kubeflowkatib/tf-mnist-with-summaries:latest

# 编写正确的 tfjob-example.yaml 
tfjob-example.yaml
https://github.com/cloudnatived/mlops/blob/main/KUBEFLOW/tfjob-example.yaml

# 查看 pod 中的多个容器的名称:
kubectl -n kubeflow-user-example-com describe pod tfjob-example-xncvxxvs-worker-0 |grep -A 10 "Containers:"

# 查看 pod 中的某个容器的日志，在分析有 sidecar 容器的 pod 时很重要:
kubectl -n kubeflow-user-example-com logs -f tfjob-example-xncvxxvs-worker-0 -c istio-validation

# 在katib-controller- 容器和 katib-ui- 容器中有 tf-job 的信息，待查。





3. Pipelines界面的提示。
-------------------------------------------------------------------------
Error: failed to retrieve list of pipelines. Click Details for more information.
......
Cannot retrieve pipeline details. Click Details for more information.
......
An error occurred
' {"error":"Failed to list pipelines in namespace kubeflow-user-example-com. Check error stack: Failed to list pipelines with context \u0026{0xc000a1a8c0}, options \u0026{10 0xc00123ee00}: 
InternalServerError: Failed to start transaction to list pipelines: Error 1049 (42000): Unknown database 'mlpipeline'","code":13,"message":"Failed to list pipelines in namespace kubeflow-user-example-com. 
Check error stack: Failed to list pipelines with context \u0026{0xc000a1a8c0}, options \u0026{10 0xc00123ee00}: InternalServerError: Failed to start transaction to list pipelines: 
Error 1049 (42000): Unknown database 'mlpipeline'","details":[{"@type":"type.googleapis.com/google.rpc.Status","code":13,"message":"Internal Server Error"}]} '
-------------------------------------------------------------------------

分析结果:
MySQL 数据库 mlpipeline不存在

# 获取 MySQL root 密码
kubectl get secret -n kubeflow mysql-secret -o jsonpath='{.data.password}' | base64 -d && echo

# 连接到 MySQL Pod
kubectl exec -it -n kubeflow deployment/mysql -- bash

# 手动在 MySQL 命令行中创建数据库：
CREATE DATABASE IF NOT EXISTS mlpipeline;
SHOW DATABASES;
EXIT;

# 或者使用一行命令创建数据库:
kubectl exec -n kubeflow deployment/mysql -- mysql -u root -p$MYSQL_ROOT_PASSWORD -e "CREATE DATABASE IF NOT EXISTS mlpipeline; SHOW DATABASES;"
kubectl exec -n kubeflow deployment/mysql -- mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS mlpipeline; SHOW DATABASES;"

# 重启 Pipelines 组件以重新初始化
# 重启 Pipelines 相关组件
kubectl rollout restart deployment -n kubeflow \
  ml-pipeline \
  ml-pipeline-persistenceagent \
  ml-pipeline-scheduledworkflow

# 等待重启完成
kubectl rollout status deployment/ml-pipeline -n kubeflow


1. 检查 Pipelines 核心组件状态
# 检查 Pipelines 相关 Pod 状态
kubectl get pods -n kubeflow | grep -E "(ml-pipeline|pipeline|cache|viewer)"

# 检查关键服务状态
kubectl get deployments -n kubeflow | grep -E "(ml-pipeline|pipeline)"

2. 查看详细错误日志
# 查看 Pipeline Server 日志
kubectl logs -n kubeflow deployment/ml-pipeline -c ml-pipeline-api-server

# 查看前端日志
kubectl logs -n kubeflow deployment/ml-pipeline-ui

# 查看 Persistent Agent 日志（如果有）
kubectl logs -n kubeflow deployment/ml-pipeline-persistenceagent

# 查看 Scheduled Workflow 日志
kubectl logs -n kubeflow deployment/ml-pipeline-scheduledworkflow





7. kuberay

https://docs.rayai.org.cn/en/latest/cluster/kubernetes/getting-started/kuberay-operator-installation.html
https://github.com/ray-project/kuberay
https://docs.rayai.org.cn/en/latest/cluster/kubernetes/getting-started/kuberay-operator-installation.html



7.1 KubeRay Operator Installation
# KubeRay Operator Installation
Step 1: Create a Kubernetes cluster

Step 2: Install KubeRay operator
kubectl create -k "github.com/ray-project/kuberay/ray-operator/config/default?ref=v1.5.0"
# 会创建以下资源:
---------------------------------------------------------------------------------------------------------------
customresourcedefinition.apiextensions.k8s.io/rayclusters.ray.io created
customresourcedefinition.apiextensions.k8s.io/rayjobs.ray.io created
customresourcedefinition.apiextensions.k8s.io/rayservices.ray.io created
serviceaccount/kuberay-operator created
role.rbac.authorization.k8s.io/kuberay-operator-leader-election created
clusterrole.rbac.authorization.k8s.io/kuberay-operator created
rolebinding.rbac.authorization.k8s.io/kuberay-operator-leader-election created
clusterrolebinding.rbac.authorization.k8s.io/kuberay-operator created
service/kuberay-operator created
deployment.apps/kuberay-operator created
---------------------------------------------------------------------------------------------------------------


Step 3: Validate Installation
kubectl get pods

rayproject/ray:2.46.0
docker.io/rayproject/ray:2.46.0
m.daocloud.io/docker.io/rayproject/ray:2.46.0
nerdctl tag m.daocloud.io/docker.io/rayproject/ray:2.46.0 rayproject/ray:2.46.0
nerdctl save rayproject/ray:2.46.0 -o ray:2.46.0
ip="10.0.10.148 10.0.10.155 10.0.10.173 10.0.10.210"; for i in $ip ; do scp ray..2.46.0 $i:/Data/IMAGES/; ssh $i "nerdctl load -i /Data/IMAGES/ray..2.46.0" ; done


7.2 RayCluster Quickstart
# RayCluster Quickstart
Step 1: Create a Kubernetes cluster

Step 2: Deploy a KubeRay operator

Step 3: Deploy a RayCluster custom resource

Step 4: Run an application on a RayCluster




7.3 RayJob Quickstart
# RayJob Quickstart
Step 1: Create a Kubernetes cluster with Kind

Step 2: Install the KubeRay operator

Step 3: Install a RayJob
kubectl apply -f https://raw.githubusercontent.com/ray-project/kuberay/v1.5.0/ray-operator/config/samples/ray-job.sample.yaml

Step 4: Verify the Kubernetes cluster status
kubectl get rayjob
kubectl get raycluster
kubectl get pods
kubectl get rayjobs.ray.io rayjob-sample -o jsonpath='{.status.jobStatus}'
kubectl get rayjobs.ray.io rayjob-sample -o jsonpath='{.status.jobDeploymentStatus}'

Step 5: Check the output of the Ray job
kubectl logs -l=job-name=rayjob-sample

Step 6: Delete the RayJob
kubectl delete -f https://raw.githubusercontent.com/ray-project/kuberay/v1.5.0/ray-operator/config/samples/ray-job.sample.yaml

Step 7: Create a RayJob with shutdownAfterJobFinishes set to true
kubectl apply -f https://raw.githubusercontent.com/ray-project/kuberay/v1.5.0/ray-operator/config/samples/ray-job.shutdown.yaml

Step 8: Check the RayJob status
# Wait until `jobStatus` is `SUCCEEDED` and `jobDeploymentStatus` is `Complete`.
kubectl get rayjobs.ray.io rayjob-sample-shutdown -o jsonpath='{.status.jobDeploymentStatus}'
kubectl get rayjobs.ray.io rayjob-sample-shutdown -o jsonpath='{.status.jobStatus}'

Step 9: Check if the KubeRay operator deletes the RayCluster
kubectl get raycluster

Step 10: Clean up
# Step 10.1: Delete the RayJob
kubectl delete -f https://raw.githubusercontent.com/ray-project/kuberay/v1.5.0/ray-operator/config/samples/ray-job.shutdown.yaml

# Step 10.2: Delete the KubeRay operator
helm uninstall kuberay-operator

# Step 10.3: Delete the Kubernetes cluster
kind delete cluster



7.4 RayService Quickstart
# RayService Quickstart
Step 1: Create a Kubernetes cluster with Kind
kind create cluster --image=kindest/node:v1.26.0

Step 2: Install the KubeRay operator

Step 3: Install a RayService
kubectl apply -f https://raw.githubusercontent.com/ray-project/kuberay/v1.5.0/ray-operator/config/samples/ray-service.sample.yaml

```
