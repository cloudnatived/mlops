# 首先下载数据集。
# wget http://download.tensorflow.org/example_images/flower_photos.tgz;
# tar zxf flower_photos.tgz

import os
from shutil import copy, rmtree
import random

def mk_file(file_path: str):
    """创建文件夹，如果已存在则删除并重建"""
    if os.path.exists(file_path):
        rmtree(file_path)  # 递归删除文件夹及其内容
    os.makedirs(file_path)  # 创建新文件夹

def main():
    # 设置随机种子，确保结果可复现
    random.seed(0)
    
    # 数据集划分比例：10%作为验证集
    split_rate = 0.1
    
    # 获取当前工作目录并构建数据集路径
    cwd = os.getcwd()
    #data_root = os.path.join(cwd, "flower_data")
    data_root = os.path.join(cwd, "./")
    origin_flower_path = os.path.join(data_root, "flower_photos")
    
    # 检查原始数据集路径是否存在
    assert os.path.exists(origin_flower_path), f"路径 '{origin_flower_path}' 不存在"
    
    # 获取所有花类别的名称（子文件夹名）
    flower_class = [cla for cla in os.listdir(origin_flower_path) 
                    if os.path.isdir(os.path.join(origin_flower_path, cla))]
    
    # 创建训练集文件夹结构
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in flower_class:
        mk_file(os.path.join(train_root, cla))  # 为每个类别创建子文件夹
    
    # 创建验证集文件夹结构
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in flower_class:
        mk_file(os.path.join(val_root, cla))  # 为每个类别创建子文件夹
    
    # 处理每个类别的图像
    for cla in flower_class:
        cla_path = os.path.join(origin_flower_path, cla)
        images = os.listdir(cla_path)  # 获取当前类别下的所有图像
        num = len(images)
        
        # 随机选择验证集图像（使用sample确保不重复）
        val_images = random.sample(images, k=int(num * split_rate))
        
        for i, image in enumerate(images):
            image_path = os.path.join(cla_path, image)
            
            if image in val_images:
                # 复制到验证集目录
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                # 复制到训练集目录
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            
            # 打印进度条（覆盖当前行实现动态更新）
            print(f"\r[{cla}] 处理中 [{i+1}/{num}]", end="")
        
        print()  # 换行，为下一个类别处理做准备
    
    print("数据划分完成！")

if __name__ == '__main__':
    main()
