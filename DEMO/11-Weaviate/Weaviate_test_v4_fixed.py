#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Weaviate_test_v4_fixed.py
import weaviate
import os
import logging
import numpy as np
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
from contextlib import closing

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_client():
    """
    初始化 Weaviate v4 客户端
    """
    try:
        client = weaviate.connect_to_custom(
            http_host=os.getenv("WEAVIATE_HOST", "172.18.6.60"),
            http_port=int(os.getenv("WEAVIATE_HTTP_PORT", "8080")),
            http_secure=os.getenv("WEAVIATE_HTTP_SECURE", "False").lower() == "true",
            grpc_host=os.getenv("WEAVIATE_GRPC_HOST", "172.18.6.60"),
            grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
            grpc_secure=os.getenv("WEAVIATE_GRPC_SECURE", "False").lower() == "true",
        )
        logger.info("✅ 成功通过自定义参数连接到 Weaviate")
        return client
    except Exception as e:
        logger.error(f"❌ 自定义连接失败: {e}")
        raise ConnectionError("无法连接到 Weaviate，请检查连接配置")

def basic_operations_example(client: weaviate.WeaviateClient):
    """
    基础操作示例：查询集合、近文本搜索
    """
    try:
        if not client.is_ready():
            logger.error("❌ Weaviate 客户端尚未就绪")
            return

        logger.info("🚀 Weaviate 客户端已就绪")
        # 获取所有集合
        collections = client.collections.list_all()
        collection_names = [col.name for col in collections]
        logger.info(f"📚 可用的集合: {collection_names}")

        # 如果有 JeopardyQuestion 集合，执行近文本搜索
        if "JeopardyQuestion" in collection_names:
            jeopardy = client.collections.get("JeopardyQuestion")
            response = jeopardy.query.near_text(
                query="science",
                limit=3
            )
            logger.info("🔍 近文本搜索结果 (JeopardyQuestion):")
            for obj in response.objects:
                logger.info(f" - {obj.properties}")
    except Exception as e:
        logger.error(f"❌ 基础操作失败: {e}")

def complete_crud_example(client: weaviate.WeaviateClient):
    """
    完整的 CRUD 操作示例：创建、读取、更新、删除
    """
    collection_name = "TestArticle"
    try:
        # 1. 确保集合不存在（如果存在先删除）
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)
            logger.info(f"ℹ️ 已删除旧的 '{collection_name}' 集合")

        # 2. 创建集合
        collection = client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.none(),  # 无需 OpenAI，使用本地向量
            properties=[
                Property(name="title", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="category", data_type=DataType.TEXT),
                Property(name="views", data_type=DataType.INT),
                Property(name="is_published", data_type=DataType.BOOL)
            ]
        )
        logger.info(f"✅ 成功创建集合: '{collection_name}'")

        # 3. 插入多样化数据
        articles = [
            {
                "title": "人工智能的未来",
                "content": "人工智能正在改变世界，应用于医疗、教育等领域。",
                "category": "科技",
                "views": 1000,
                "is_published": True
            },
            {
                "title": "量子计算简介",
                "content": "量子计算利用量子力学原理，提供超高计算能力。",
                "category": "科技",
                "views": 500,
                "is_published": False
            },
            {
                "title": "绿色能源的挑战",
                "content": "可再生能源面临成本和技术瓶颈。",
                "category": "环境",
                "views": 750,
                "is_published": True
            }
        ]
        article_ids = []
        for article in articles:
            article_id = collection.data.insert(properties=article)
            article_ids.append(article_id)
            logger.info(f"✅ 数据插入成功, ID: {article_id}")

        # 4. 查询数据
        response = collection.query.fetch_objects(limit=5)
        logger.info("📄 所有文章:")
        for obj in response.objects:
            logger.info(f" - {obj.properties}")

        # 5. 更新数据
        article_id = article_ids[0]
        collection.data.update(
            uuid=article_id,
            properties={"views": 1500, "is_published": False}
        )
        logger.info(f"✅ 更新文章 ID: {article_id}")

        # 6. 精确匹配查询
        response = collection.query.fetch_objects(
            filters=Filter.by_property("category").equal("科技"),
            limit=2
        )
        logger.info("🔍 精确匹配查询结果 (category=科技):")
        for obj in response.objects:
            logger.info(f" - {obj.properties}")

        # 7. 混合搜索（向量 + 过滤）
        response = collection.query.hybrid(
            query="人工智能",
            alpha=0.7,  # 平衡向量和关键字搜索
            filters=Filter.by_property("is_published").equal(True),
            limit=2
        )
        logger.info("🔍 混合搜索结果 (人工智能 + 已发布):")
        for obj in response.objects:
            logger.info(f" - {obj.properties}")

        # 8. 聚合查询
        response = collection.aggregate.over_all(
            group_by=Filter.by_property("category"),
            total_count=True
        )
        logger.info("📊 按类别聚合:")
        for group in response.groups:
            logger.info(f" - 类别: {group.grouped_by.value}, 计数: {group.total_count}")

        # 9. 删除集合
        client.collections.delete(collection_name)
        logger.info(f"✅ 成功删除集合: '{collection_name}'")

    except Exception as e:
        logger.error(f"❌ CRUD 操作失败: {e}")

def batch_operations_example(client: weaviate.WeaviateClient):
    """
    批量操作示例：批量插入、批量删除
    """
    collection_name = "TestBatch"
    try:
        # 1. 确保集合存在
        if not client.collections.exists(collection_name):
            client.collections.create(
                name=collection_name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="index", data_type=DataType.INT)
                ]
            )
            logger.info(f"✅ 成功创建集合: '{collection_name}'")

        collection = client.collections.get(collection_name)

        # 2. 批量插入
        with collection.batch.dynamic() as batch:
            for i in range(20):
                batch.add_object(
                    properties={"title": f"批量文章 {i+1}", "index": i+1}
                )
        logger.info("✅ 批量插入完成")

        # 3. 验证插入
        response = collection.aggregate.over_all(total_count=True)
        logger.info(f"📊 验证集合总对象数: {response.total_count}")

        # 4. 批量删除
        collection.data.delete_many(
            where=Filter.by_property("index").greater_than(10)
        )
        logger.info("✅ 删除 index > 10 的对象")

        # 5. 验证删除
        response = collection.aggregate.over_all(total_count=True)
        logger.info(f"📊 删除后集合总对象数: {response.total_count}")

        # 6. 删除集合
        client.collections.delete(collection_name)
        logger.info(f"✅ 成功删除集合: '{collection_name}'")

    except Exception as e:
        logger.error(f"❌ 批量操作失败: {e}")

def vector_operations_example(client: weaviate.WeaviateClient):
    """
    向量操作示例：自定义向量、向量搜索
    """
    collection_name = "TestVector"
    try:
        # 1. 创建带自定义向量的集合
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)

        client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.none(),  # 使用自定义向量
            properties=[
                Property(name="name", data_type=DataType.TEXT),
                Property(name="score", data_type=DataType.NUMBER)
            ]
        )
        logger.info(f"✅ 成功创建集合: '{collection_name}'")

        collection = client.collections.get(collection_name)

        # 2. 插入带自定义向量的数据
        objects = [
            {
                "properties": {"name": "对象A", "score": 0.8},
                "vector": np.random.rand(128).tolist()  # 随机 128 维向量
            },
            {
                "properties": {"name": "对象B", "score": 0.9},
                "vector": np.random.rand(128).tolist()
            },
            {
                "properties": {"name": "对象C", "score": 0.7},
                "vector": np.random.rand(128).tolist()
            }
        ]
        for obj in objects:
            collection.data.insert(
                properties=obj["properties"],
                vector=obj["vector"]
            )
        logger.info("✅ 自定义向量插入完成")

        # 3. 向量搜索
        query_vector = np.random.rand(128).tolist()
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=2
        )
        logger.info("🔍 向量搜索结果:")
        for obj in response.objects:
            logger.info(f" - {obj.properties}")

        # 4. 删除集合
        client.collections.delete(collection_name)
        logger.info(f"✅ 成功删除集合: '{collection_name}'")

    except Exception as e:
        logger.error(f"❌ 向量操作失败: {e}")

if __name__ == "__main__":
    client = None
    try:
        # 初始化客户端
        client = init_client()
        logger.info("🎉 Weaviate v4 客户端初始化成功!")

        # 运行示例
        logger.info("\n" + "="*50)
        logger.info("运行基础操作示例:")
        basic_operations_example(client)

        logger.info("\n" + "="*50)
        logger.info("运行完整 CRUD 示例:")
        complete_crud_example(client)

        logger.info("\n" + "="*50)
        logger.info("运行批量操作示例:")
        batch_operations_example(client)

        logger.info("\n" + "="*50)
        logger.info("运行向量操作示例:")
        vector_operations_example(client)

    except Exception as e:
        logger.error(f"❌ 主程序执行失败: {e}")
    finally:
        if client:
            client.close()
            logger.info("\n🚪 客户端连接已关闭。")
