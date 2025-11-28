#!/bin/bash
# deploy_flux2.sh

# 设置共享存储
SHARED_DIR="/shared"
mkdir -p $SHARED_DIR/{models,logs}

# 下载FLUX-2模型权重
cd $SHARED_DIR/models
git lfs install
git clone https://huggingface.co/black-forest-labs/FLUX.1-dev flux2

# 创建模型目录结构
cd flux2
mkdir -p 1
mv config.pbtxt model.py 1/

# 启动Triton集群
docker-compose up -d

# 等待服务就绪
echo "等待Triton服务器启动..."
sleep 30

# 检查服务状态
for port in 8000 8003 8006; do
    curl -f http://localhost:$port/v2/health/ready
    if [ $? -eq 0 ]; then
        echo "Triton服务在端口 $port 已就绪"
    else
        echo "Triton服务在端口 $port 启动失败"
    fi
done

echo "FLUX-2部署完成！访问地址: http://localhost:8080"
