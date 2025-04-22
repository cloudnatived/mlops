kubespray-2.27.0 部署笔记

### 4.1.kubespray-2.27.0 部署笔记

4.1.kubespray-2.27.0

```text
kubespary:
https://github.com/kubernetes-sigs/kubespray/archive/refs/tags/v2.27.0.tar.gz

ubuntu-22.04.5-live-server-amd64.iso
ubuntu-24.10-live-server-amd64.iso  #完成测试，但是不习惯python的管理工具
```

设置基础环境

```text
sed -i 's/SELINUX=enforcing/SELINUX=disabled/g' /etc/selinux/config;
sed -i 's/GSSAPIAuthentication yes/GSSAPIAuthentication no/g' /etc/ssh/sshd_config;
echo "GSSAPIAuthentication no" >> /etc/ssh/sshd_config;
sed -i 's/UseDNS yes/UseDNS no/g' /etc/ssh/sshd_config;
cat >> /etc/ssh/sshd_config <<EOF
UseDNS no
PermitRootLogin yes
EOF

systemctl restart sshd;

cp /etc/apt/sources.list /etc/apt/sources.list.original;
cat > /etc/apt/sources.list <<EOF
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-proposed main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-proposed main restricted universe multiverse
EOF

apt update -y;
apt list --upgradable;
apt upgrade -y;

# 为了解决这个WARNING：
#########################################################
root@x:~# netplan apply
WARNING:root:Cannot call Open vSwitch: ovsdb-server.service is not running.
#########################################################
chmod 600 /etc/netplan/*
touch /etc/cloud/cloud-init.disabled;
apt -y install openvswitch-switch;
systemctl disable openvswitch-switch.service;

# 运行在init 3
systemctl isolate multi-user.target;
systemctl isolate runlevel3.target;
ln -sf /lib/systemd/system/multi-user.target /etc/systemd/system/default.target;
systemctl set-default multi-user.target;

# 关闭不需要的服务：
systemctl list-unit-files |awk '{ print $1,$2 }'|grep enable|egrep -v "ssh|multi|systemd-resolved|wpa_" |awk '{ print $1}'|xargs -i systemctl disable {};

# 确认服务已关闭：
systemctl list-unit-files |awk '{print $1,$2}'|grep enabled;

apt install -y python3-pip python3 python3-netaddr wget git;
apt install -y python3-dev;
pip install --upgrade pip;

pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple;
mkdir -p /root/.config/pip;
cat > /root/.config/pip/pip.conf <<EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
EOF

# 以下配置host及环境：
cat > /etc/hostname <<EOF
k101
EOF

cat >> /etc/hosts <<EOF
192.168.32.101 k101
EOF

hostname k101;

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

net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1

fs.inotify.max_user_instances=2280
fs.inotify.max_user_watches=655360
EOF

ssh-keygen -t rsa -N "";
cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys;
```



下载和修改kubespary，安装

```text
wget https://github.com/kubernetes-sigs/kubespray/archive/refs/tags/v2.27.0.tar.gz

#安装 kubespray 依赖
cd /opt/kubespray-2.27.0;

# 安装依赖
pip3 install -r requirements.txt;

# 复制一份 自己的配置
cd /opt/kubespray;
cp -au /opt/kubespray-2.24.1/inventory/sample /opt/kubespray-2.27.0/inventory/bbc;

# 修改配置 hosts.yaml 是这样部署了。
root@node1:/opt/kubespray-2.27.0/inventory/bbc# cat hosts.yaml
all:
  hosts:
    node1:
      ansible_host: y133
      ip: 10.0.10.133
      access_ip: 10.0.10.133
    node2:
      ansible_host: y134
      ip: 10.0.10.134
      access_ip: 10.0.10.134
    node3:
      ansible_host: y135
      ip: 10.0.10.135
      access_ip: 10.0.10.135
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
#########################################################

vim /opt/kubespray-2.27.0/inventory/bbc/group_vars/all/all.yml;
#########################################################
# 打开下面这个选项:
loadbalancer_apiserver_localhost: true
#########################################################

# 修改如下配置，海外的云服务器不要设置这两个镜像，海外的云服务器不要设置这两个镜像。: 
vim /opt/kubespray-2.27.0/inventory/bbc/group_vars/all/docker.yml;
#########################################################+
docker_registry_mirrors:
  - https://registry.docker-cn.com
  - https://mirror.aliyuncs.com
#########################################################

# 以下两段都需要注释掉：
vim /opt/kubespray-2.27.0/extra_playbooks/roles/kubernetes/preinstall/tasks/0040-verify-settings.yml;
#########################################################
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
#########################################################  

# 如果是云服务器，应该关闭内存检查。把以下内容全部注释掉，使其不生效。
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
#########################################################

cat > /opt/kubespray-2.27.0/inventory/bbc/group_vars/all/mirror.yml <<EOF
gcr_image_repo: "gcr.m.daocloud.io"
kube_image_repo: "k8s.m.daocloud.io"
docker_image_repo: "docker.m.daocloud.io"
quay_image_repo: "quay.m.daocloud.io"
github_image_repo: "ghcr.m.daocloud.io"
files_repo: "https://files.m.daocloud.io"
EOF

cat  > /opt/kubespray-2.27.0/inventory/bbc/group_vars/etcd.yml <<EOF
etcd_deployment_type: host
EOF

vim /opt/kubespray-2.27.0/inventory/bbc/group_vars/k8s_cluster/k8s-cluster.yml
auto_renew_certificates: true

# 可选，如果需要设置代理：
vim /opt/kubespray-2.27.0/extra_playbooks/inventory/bbc/group_vars/all/all.yml; 
http_proxy: "http://192.168.1.5:7890"
https_proxy: "http://192.168.1.5:7890"
no_proxy: "http://localhost:8080/,192.168.*.*,*.local,*.localhost*,localhost,127.0.0.1,192.168.1.100"

#########################################################
export http_proxy=http://192.168.1.5:7890;
export https_proxy=https://192.168.1.5:7890;
#########################################################

# kubespray-2.27.0 
cd /opt/kubespray-2.27.0/;
# 使用cilium网络，不推荐，可能会造成重启网络时，网络协议栈无法完成重启。
ansible-playbook -i inventory/bbc/hosts.yaml --become --become-user=root -e kube_network_plugin=cilium cluster.yml

# 使用flannel网络。
ansible-playbook -i inventory/bbc/hosts.yaml --become --become-user=root -e kube_network_plugin=flannel cluster.yml

# 重新安装，增加1个计算节点，重新运行安装命令。
ansible-playbook -i inventory/bbc/hosts.yaml --become --become-user=root -e kube_network_plugin=flannel cluster.yml
```



安装过程中可能会碰到的问题：

```text
# 错误提示：bridge-nf-call-iptables: No such file or directory
# 错误提示：bridge-nf-call-iptables: No such file or directory\nsysctl: cannot stat
# 需要加载模块。
modprobe br_netfilter;

# 报错，
-----------------------
TASK [container-engine/containerd : Download_file | Validate mirrors] **************************************************************************************************
failed: [k100] (item=None) => {"attempts": 4, "censored": "the output has been hidden due to the fact that 'no_log: true' was specified for this result", "changed": false}
fatal: [k100 -> {{ download_delegate if download_force_cache else inventory_hostname }}]: FAILED! => {"censored": "the output has been hidden due to the fact that 'no_log: true' was specified for this result", "changed": false}
...ignoring
Thursday 11 April 2024  12:14:06 +0000 (0:03:22.953)       0:05:55.774 ******** 

TASK [container-engine/containerd : Download_file | Get the list of working mirrors] ***********************************************************************************
ok: [k100]
Thursday 11 April 2024  12:14:07 +0000 (0:00:00.591)       0:05:56.365 ******** 

TASK [container-engine/containerd : Download_file | Download item] *****************************************************************************************************
fatal: [k100]: FAILED! => {"censored": "the output has been hidden due to the fact that 'no_log: true' was specified for this result", "changed": false}
-----------------------

# 23.1 还会出这个问题。
# 需要启动这个服务。要启动resolved服务：
systemctl enable systemd-resolved.service;
systemctl restart systemd-resolved.service;

# 需要enabled的系统服务：
root@kubernetes-1:~# systemctl list-unit-files |awk '{print $1,$2}'|grep enabled;
containerd.service enabled
etcd.service enabled
kubelet.service enabled
multipathd.service enabled
rc-local.service enabled
ssh.service enabled
systemd-fsck-root.service enabled-runtime #关闭会导致分区只读挂载
systemd-networkd.service enabled-runtime
systemd-remount-fs.service enabled-runtime #关闭会导致分区只读挂载
systemd-resolved.service enabled
systemd-timesyncd.service enabled
multipathd.socket enabled
```



部署完之后：

```text
#检查启动的服务：
root@kubernetes-101:/opt/kubespray-2.23.1# systemctl list-unit-files |awk '{ print $1,$2 }'|grep enable
containerd.service enabled
etcd.service enabled
kubelet.service enabled
multipathd.service enabled
netplan-ovs-cleanup.service enabled-runtime
ssh.service enabled
systemd-fsck-root.service enabled-runtime
systemd-networkd-wait-online.service enabled-runtime
systemd-networkd.service enabled-runtime
systemd-remount-fs.service enabled-runtime
systemd-resolved.service enabled
multipathd.socket enabled
systemd-networkd.socket enabled

#关闭代理
vim /etc/apt/apt.conf;
vim /etc/systemd/system/containerd.service.d/http-proxy.conf;
```



kubernetes优化
参考文档：
大规模场景下 kubernetes 集群的性能优化    https://zhuanlan.zhihu.com/p/111244925
Kubernetes：k8s优化大法（江湖失传已久的武林秘籍）    https://www.cnblogs.com/unqiang/p/18360801
17个应该了解的Kubernetes优化    https://cloud.tencent.com/developer/article/2402219
如何优化Kubernetes的性能和资源利用率优化    https://blog.csdn.net/u010349629/article/details/130638445
Kubernetes各组件参数配置优化建议    https://blog.csdn.net/ywq935/article/details/103124541
