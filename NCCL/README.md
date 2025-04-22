
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
