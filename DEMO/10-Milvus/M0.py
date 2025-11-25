#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# M.py
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import numpy as np
from pymilvus import utility
import random
from sentence_transformers import SentenceTransformer  # 用于文本嵌入
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch import nn
import torchvision

connections.connect("default", host="172.18.6.60", port="19530")
def collection_exists(collection_name):
    return collection_name in utility.list_collections()
 
if collection_exists("multimodal_collection"):
    col = Collection(name="multimodal_collection")
    col.drop()

# 3. 定义集合字段并创建集合：
# 功能：定义了一个集合，包含 id（主键）、text_vector（文本向量，384 维）、tabular_vector（表格向量，128 维）、image_vector（图片向量，2048 维）和 metadata（元数据，字符串类型）。

# 4. 数据处理和插入：
# 文本数据处理（使用 Sentence Transformers）
def get_text_embedding(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(text)
    return embedding.astype(np.float32)
 
# 表格数据处理（模拟：随机向量）
def get_tabular_embedding(tabular_data):
    return np.random.random(128).astype(np.float32)
 
# 图片数据处理（使用预训练的 CNN 模型）
def get_image_embedding(image_path):
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.fc = Identity()
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(image_tensor)
    return embedding.squeeze().numpy().astype(np.float32)
 
# 示例数据
texts = ["这是一个文本示例。", "这是另一个文本示例。"]
tabular_data = [{"feature1": 0.5, "feature2": 0.3}, {"feature1": 0.7, "feature2": 0.9}]
image_paths = ["./data/page_4.png", "./data/page_8.png"]
 
# 处理示例数据
text_embeddings = [get_text_embedding(text) for text in texts]
tabular_embeddings = [get_tabular_embedding(data) for data in tabular_data]
image_embeddings = [get_image_embedding(path) for path in image_paths]
ids = [1, 2]
metadata = ["text and image data 1", "text and image data 2"]
 
# 将数据插入到集合
collection.insert([ids, text_embeddings, tabular_embeddings, image_embeddings, metadata])
collection.flush()  # 确保数据写入

# 5. 创建索引：
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 1024}
}
 
collection.create_index(field_name="text_vector", index_params=index_params)
collection.create_index(field_name="tabular_vector", index_params=index_params)
collection.create_index(field_name="image_vector", index_params=index_params)
print("Indexes created.")

# 6. 加载集合：
collection.load()
print(f"Collection loaded with {collection.num_entities} entities.")

# 7. 相似性搜索（基于文本向量）
search_text_embedding = get_text_embedding("这是一个查询文本。").reshape(1, -1)
results = collection.search(search_text_embedding, "text_vector", {"metric_type": "L2", "params": {"nprobe": 10}}, limit=2)
print("Text search results:")
for result in results:
    for r in result:
        print(f"ID: {r.id}, Distance: {r.distance}")
