Ubuntu 24.04.3 LTS部署OpenStack E版
ubuntu-24.04.3-live-server-amd64.iso

```
ubuntu-24.04.3-live-server-amd64.iso

# 安装完操作系统后，配置IP。
ip link set enp4s1 up
ip addr add 172.18.8.209/24 dev enp4s1
ip route add default via 172.18.8.1

# 安装完操作系统后，配置IP。
ip link set enp4s1 up
ip addr add 172.18.8.209/24 dev enp4s1
ip route add default via 172.18.8.1

# 写入网卡配置文件。
cat > /etc/netplan/50-cloud-init.yaml <<EOF
# network: {config: disabled}
network:
  ethernets:
    enp2s0f0: # 网卡名可能有区别
      dhcp4: false
      addresses:
        - 172.18.6.59/24
      routes:
        - to: default
          via: 172.18.6.1
      nameservers:
         addresses: [8.8.8.8]
  version: 2
EOF

# 使用netplan应用网络配置。
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

apt install -y python3-pip python3 python3-netaddr wget git;
apt install -y python3-dev;
pip install --upgrade pip;

# 解决ubuntu24.04 使用pip3时的信息提醒：error: externally-managed-environment
mv /usr/lib/python3.12/EXTERNALLY-MANAGED /usr/lib/python3.12/EXTERNALLY-MANAGED.bk

pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple;
mkdir -p /root/.config/pip;
cat > /root/.config/pip/pip.conf <<EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
EOF

uname -r         # 查看内核版本
lsb_release -a   # 查看发行版信息（Ubuntu/Debian/CentOS等）

# 确认安装了gcc make
apt install -y gcc make g++ net-tools

# 重启一次。
reboot


# 使用清华的docker-ce的源。
curl -fsSL https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository -y "deb [arch=amd64] https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu $(lsb_release -cs) stable"

sudo apt-get install -y docker-ce docker-ce-cli containerd.io # 先安装这些。


临时禁用Swap（OpenStack要求）
sudo swapoff -a
# 永久禁用Swap（注释/etc/fstab中的Swap行）
sudo sed -i '/swap/s/^/#/' /etc/fstab


https://opendev.org/openstack/kolla-ansible/src/branch/stable/2025.1/


git clone https://github.com/openstack/kolla-ansible.git

root@anhua69:/etc/kolla# cat globals.yml |grep -v ^#|grep -v ^$
---
workaround_ansible_issue_8743: yes
kolla_base_distro: "ubuntu"
kolla_internal_vip_address: "10.10.10.254"
network_interface: "enp2s0f0"


root@anhua69:/opt/kolla# cat multinode 
[control]
anhua69
anhua70
anhua71

[network]
anhua69       
anhua70       
anhua71

[compute]
anhua69       
anhua70       
anhua71

[monitoring]
anhua69       
anhua70       
anhua71

[storage]
anhua69       
anhua70       
anhua71


拉取Docker镜像：
kolla-ansible pull
部署所有服务：
kolla-ansible deploy

```







