#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# M1.py - 兼容多版本PyTorch的Milvus多模态示例
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import numpy as np
from pymilvus import utility
import random
import os
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. 定义Identity类（用于替换ResNet的全连接层）
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

# 2. 兼容性导入和版本检查
def setup_environment():
    """设置环境，处理不同版本的PyTorch和模型加载"""
    
    # 检查PyTorch是否可用
    try:
        import torch
        import torchvision
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"Torchvision版本: {torchvision.__version__}")
    except ImportError as e:
        logger.error("未安装PyTorch，请先安装: pip install torch torchvision")
        sys.exit(1)
    
    # 尝试导入sentence-transformers，如果失败则使用备用方案
    text_encoder = None
    try:
        from sentence_transformers import SentenceTransformer
        text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ 成功加载sentence-transformers")
    except ImportError as e:
        logger.warning("❌ 无法导入sentence-transformers，将使用备用文本编码方案")
        logger.warning(f"错误信息: {e}")
    except Exception as e:
        logger.warning(f"❌ sentence-transformers加载失败: {e}，使用备用方案")
    
    return text_encoder

# 3. 备用文本编码方案
class BackupTextEncoder:
    """当sentence-transformers不可用时使用的备用文本编码器"""
    
    def __init__(self, dim=384):
        self.dim = dim
        logger.info(f"使用备用文本编码器，维度: {dim}")
    
    def encode(self, text):
        """生成随机文本向量（生产环境中应替换为更合理的编码方法）"""
        # 这里使用基于文本哈希的确定性随机向量
        import hashlib
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        return rng.randn(self.dim).astype(np.float32)

# 4. 改进的文本嵌入函数
def get_text_embedding(text, text_encoder=None):
    """兼容多种情况的文本嵌入函数"""
    if text_encoder is not None:
        try:
            # 使用sentence-transformers
            embedding = text_encoder.encode(text)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.warning(f"sentence-transformers编码失败: {e}，使用备用方案")
    
    # 使用备用编码器
    backup_encoder = BackupTextEncoder()
    return backup_encoder.encode(text)

# 5. 表格数据处理（改进版）
def get_tabular_embedding(tabular_data, method="random"):
    """生成表格数据向量，支持多种方法"""
    if method == "random":
        # 随机向量（默认）
        return np.random.random(128).astype(np.float32)
    elif method == "hash_based":
        # 基于数据哈希的确定性向量
        import hashlib
        data_str = str(tabular_data)
        seed = int(hashlib.md5(data_str.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        return rng.randn(128).astype(np.float32)
    else:
        return np.random.random(128).astype(np.float32)

# 6. 改进的图片嵌入函数（兼容不同PyTorch版本）
def get_image_embedding(image_path, model=None):
    """兼容不同PyTorch版本的图片嵌入函数"""
    try:
        import torch
        from PIL import Image
        import torchvision.transforms as transforms
        import torchvision
        
        if model is None:
            # 动态加载模型，兼容不同版本的torchvision
            try:
                # 尝试使用新版本的weights参数
                if hasattr(torchvision.models, 'ResNet50_Weights'):
                    weights = torchvision.models.ResNet50_Weights.DEFAULT
                    model = torchvision.models.resnet50(weights=weights)
                else:
                    # 旧版本兼容
                    model = torchvision.models.resnet50(pretrained=True)
            except Exception as e:
                logger.warning(f"ResNet50加载失败: {e}，使用随机初始化")
                model = torchvision.models.resnet50(pretrained=False)
        
        # 替换全连接层
        model.fc = Identity()
        model.eval()
        
        # 图像预处理（兼容不同版本）
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 加载和处理图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"图像加载失败 {image_path}: {e}，使用随机向量")
            return np.random.random(2048).astype(np.float32)
        
        image_tensor = transform(image).unsqueeze(0)
        
        # 使用GPU如果可用
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            embedding = model(image_tensor)
        
        return embedding.cpu().squeeze().numpy().astype(np.float32)
        
    except Exception as e:
        logger.error(f"图片处理失败: {e}")
        # 返回随机向量作为备用
        return np.random.random(2048).astype(np.float32)

# 7. 备用图片嵌入函数
def get_image_embedding_backup(image_path, dim=2048):
    """当主要方法失败时使用的备用图片嵌入"""
    logger.info(f"使用备用图片编码方案: {image_path}")
    # 基于文件路径生成确定性随机向量
    import hashlib
    seed = int(hashlib.md5(image_path.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    return rng.randn(dim).astype(np.float32)

# 8. 创建集合模式
def create_collection_schema():
    """创建多模态集合的模式"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="tabular_vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=2048),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=500)
    ]
    
    schema = CollectionSchema(fields, "多模态数据集合")
    return schema

# 9. 检查集合是否存在
def collection_exists(collection_name):
    """检查集合是否存在"""
    try:
        return collection_name in utility.list_collections()
    except Exception as e:
        logger.error(f"检查集合存在性失败: {e}")
        return False

# 10. 主程序
def main():
    """主程序入口"""
    logger.info("🚀 开始多模态Milvus示例程序")
    
    # 设置环境
    text_encoder = setup_environment()
    
    # 连接到Milvus
    try:
        connections.connect("default", host="172.18.6.60", port="19530")
        logger.info("✅ 成功连接到Milvus")
    except Exception as e:
        logger.error(f"❌ 连接Milvus失败: {e}")
        return
    
    # 集合名称
    collection_name = "multimodal_collection"
    
    # 清理现有集合（如果存在）
    if collection_exists(collection_name):
        try:
            col = Collection(name=collection_name)
            col.drop()
            logger.info(f"✅ 已删除现有集合: {collection_name}")
        except Exception as e:
            logger.error(f"❌ 删除集合失败: {e}")
            return
    
    # 创建新集合
    try:
        schema = create_collection_schema()
        collection = Collection(name=collection_name, schema=schema)
        logger.info(f"✅ 成功创建集合: {collection_name}")
    except Exception as e:
        logger.error(f"❌ 创建集合失败: {e}")
        return
    
    # 准备示例数据
    texts = [
        "这是一个文本示例，展示人工智能的应用。",
        "这是另一个文本示例，讨论机器学习的未来。",
        "第三个示例文本，关于深度学习和神经网络。"
    ]
    
    tabular_data = [
        {"feature1": 0.5, "feature2": 0.3, "category": "A"},
        {"feature1": 0.7, "feature2": 0.9, "category": "B"},
        {"feature1": 0.2, "feature2": 0.8, "category": "A"}
    ]
    
    # 图像路径 - 如果文件不存在，将使用备用方案
    image_paths = [
        "./data/page_4.png",
        "./data/page_8.png", 
        "./data/sample_image.jpg"  # 备用路径
    ]
    
    ids = [1, 2, 3]
    metadata = [
        "文本和图像数据示例1",
        "文本和图像数据示例2", 
        "文本和图像数据示例3"
    ]
    
    # 处理数据
    logger.info("📊 开始处理多模态数据...")
    
    text_embeddings = []
    tabular_embeddings = []
    image_embeddings = []
    
    for i, text in enumerate(texts):
        # 文本嵌入
        text_embedding = get_text_embedding(text, text_encoder)
        text_embeddings.append(text_embedding)
        
        # 表格嵌入
        tabular_embedding = get_tabular_embedding(tabular_data[i], method="hash_based")
        tabular_embeddings.append(tabular_embedding)
        
        # 图像嵌入（带错误处理）
        image_path = image_paths[i] if i < len(image_paths) else image_paths[0]
        if os.path.exists(image_path):
            try:
                image_embedding = get_image_embedding(image_path)
            except Exception as e:
                logger.warning(f"主要图片编码失败，使用备用方案: {e}")
                image_embedding = get_image_embedding_backup(image_path)
        else:
            logger.warning(f"图像文件不存在: {image_path}，使用备用方案")
            image_embedding = get_image_embedding_backup(image_path)
        
        image_embeddings.append(image_embedding)
        
        logger.info(f"✅ 处理完成第 {i+1} 条数据")
    
    # 插入数据到集合
    try:
        collection.insert([ids, text_embeddings, tabular_embeddings, image_embeddings, metadata])
        collection.flush()
        logger.info(f"✅ 成功插入 {len(ids)} 条数据")
    except Exception as e:
        logger.error(f"❌ 数据插入失败: {e}")
        return
    
    # 创建索引
    try:
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024}
        }
        
        collection.create_index(field_name="text_vector", index_params=index_params)
        collection.create_index(field_name="tabular_vector", index_params=index_params)
        collection.create_index(field_name="image_vector", index_params=index_params)
        logger.info("✅ 索引创建成功")
    except Exception as e:
        logger.error(f"❌ 索引创建失败: {e}")
        return
    
    # 加载集合
    try:
        collection.load()
        logger.info(f"✅ 集合加载成功，实体数量: {collection.num_entities}")
    except Exception as e:
        logger.error(f"❌ 集合加载失败: {e}")
        return
    
    # 执行搜索测试
    logger.info("🔍 开始搜索测试...")
    
    # 文本搜索
    try:
        search_text = "人工智能"
        search_embedding = get_text_embedding(search_text, text_encoder).reshape(1, -1)
        
        results = collection.search(
            search_embedding, 
            "text_vector", 
            {"metric_type": "L2", "params": {"nprobe": 10}}, 
            limit=2,
            output_fields=["metadata"]
        )
        
        logger.info("📝 文本搜索结果:")
        for i, hit in enumerate(results[0]):
            logger.info(f"  排名 {i+1}: ID={hit.id}, 距离={hit.distance:.4f}, 元数据={hit.entity.get('metadata')}")
    except Exception as e:
        logger.error(f"❌ 文本搜索失败: {e}")
    
    # 图像搜索（使用第一条数据的图像向量）
    try:
        if len(image_embeddings) > 0:
            search_image_embedding = np.array([image_embeddings[0]])
            
            results = collection.search(
                search_image_embedding,
                "image_vector",
                {"metric_type": "L2", "params": {"nprobe": 10}},
                limit=2,
                output_fields=["metadata"]
            )
            
            logger.info("🖼️ 图像搜索结果:")
            for i, hit in enumerate(results[0]):
                logger.info(f"  排名 {i+1}: ID={hit.id}, 距离={hit.distance:.4f}, 元数据={hit.entity.get('metadata')}")
    except Exception as e:
        logger.error(f"❌ 图像搜索失败: {e}")
    
    # 性能统计
    logger.info("📈 性能统计:")
    logger.info(f"  - 文本向量维度: {len(text_embeddings[0]) if text_embeddings else 'N/A'}")
    logger.info(f"  - 表格向量维度: {len(tabular_embeddings[0]) if tabular_embeddings else 'N/A'}")
    logger.info(f"  - 图像向量维度: {len(image_embeddings[0]) if image_embeddings else 'N/A'}")
    logger.info(f"  - 总数据量: {collection.num_entities}")
    
    logger.info("🎉 程序执行完成！")

if __name__ == "__main__":
    # 添加torch导入（在文件顶部已定义，这里确保可用）
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        # 如果torch不可用，定义空的nn模块
        class nn:
            class Module:
                pass
        
        logger.warning("PyTorch nn模块不可用，使用简化版本")
    
    main()
