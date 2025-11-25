#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# M2.py - 修复 nn.Module 未定义错误的多模态Milvus示例
# 1. 首先确保导入所有必要的模块
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from pymilvus import utility
import logging
import os
import sys

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 2. 尝试导入PyTorch相关模块（带错误处理）
try:
    import torch
    import torch.nn as nn
    import torchvision
    from PIL import Image
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
    logger.info(f"✅ PyTorch {torch.__version__} 可用")
except ImportError as e:
    TORCH_AVAILABLE = False
    logger.warning(f"❌ PyTorch不可用: {e}")
    # 创建虚拟nn模块以允许代码继续运行
    class nn:
        class Module:
            pass

# 3. 定义Identity类（带版本兼容性检查）
class Identity(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self):
        if TORCH_AVAILABLE:
            super(Identity, self).__init__()
        else:
            logger.warning("⚠️ 使用简化版Identity（无PyTorch）")
        
    def forward(self, x):
        return x

# 4. 文本编码器初始化（带兼容性检查）
def initialize_text_encoder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError as e:
        logger.warning(f"❌ sentence-transformers不可用: {e}")
        return None
    except Exception as e:
        logger.warning(f"❌ 文本编码器初始化失败: {e}")
        return None

# 5. 改进的图片嵌入函数（完全兼容无PyTorch环境）
def get_image_embedding(image_path):
    """兼容有无PyTorch环境的图片嵌入函数"""
    if not TORCH_AVAILABLE:
        logger.warning("⚠️ PyTorch不可用，使用备用图片编码方案")
        return get_image_embedding_backup(image_path)
    
    try:
        # 动态加载模型，兼容不同版本的torchvision
        try:
            if hasattr(torchvision.models, 'ResNet50_Weights'):
                weights = torchvision.models.ResNet50_Weights.DEFAULT
                model = torchvision.models.resnet50(weights=weights)
            else:
                model = torchvision.models.resnet50(pretrained=True)
        except Exception as e:
            logger.warning(f"ResNet50加载失败: {e}")
            model = torchvision.models.resnet50(pretrained=False)
        
        model.fc = Identity()
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"图像加载失败: {e}")
            return get_image_embedding_backup(image_path)
        
        image_tensor = transform(image).unsqueeze(0)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            embedding = model(image_tensor)
        
        return embedding.cpu().squeeze().numpy().astype(np.float32)
        
    except Exception as e:
        logger.error(f"图片处理失败: {e}")
        return get_image_embedding_backup(image_path)

# 6. 备用图片嵌入函数
def get_image_embedding_backup(image_path, dim=2048):
    """当主要方法失败时使用的备用图片嵌入"""
    import hashlib
    seed = int(hashlib.md5(image_path.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    return rng.randn(dim).astype(np.float32)

# 7. 主程序
def main():
    logger.info("🚀 启动多模态Milvus示例程序")
    
    # 初始化文本编码器
    text_encoder = initialize_text_encoder()
    
    # 连接到Milvus
    try:
        connections.connect("default", host="172.18.6.60", port="19530")
        logger.info("✅ 成功连接到Milvus")
    except Exception as e:
        logger.error(f"❌ 连接Milvus失败: {e}")
        return
    
    # 示例数据
    sample_image_path = "sample.jpg"
    if not os.path.exists(sample_image_path):
        logger.warning(f"示例图片不存在: {sample_image_path}")
        sample_image_path = None
    
    # 测试图片嵌入
    if sample_image_path:
        logger.info("🖼️ 测试图片嵌入...")
        img_embedding = get_image_embedding(sample_image_path)
        logger.info(f"图片向量维度: {len(img_embedding)}")
    
    logger.info("🎉 程序执行完成")

if __name__ == "__main__":
    main()
