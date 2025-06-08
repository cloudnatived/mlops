import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    # ---------------------- 1. 配置参数与路径 ----------------------
    # 模型路径（需替换为实际路径）
    model_path = "./Lenet.pth"  # 假设模型权重保存于此
    # 数据集路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))
    # CIFAR10类别名称（用于可视化）
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # ---------------------- 2. 数据预处理 ----------------------
    # 测试集预处理（需与训练时一致）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # ---------------------- 3. 加载测试数据集 ----------------------
    # 加载CIFAR10测试集（10000张图像）
    test_set = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        #download=True,  # 首次运行设为True，下载后改为False
        download=False,  # 首次运行设为True，下载后改为False
        transform=transform
    )
    
    # 构建测试数据加载器（批次大小可根据内存调整）
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=64,
        shuffle=False,
        num_workers=0  # Windows环境设为0
    )
    
    # ---------------------- 4. 加载模型 ----------------------
    from model import LeNet  # 导入模型定义
    
    # 实例化模型
    net = LeNet()
    # 加载模型权重
    net.load_state_dict(torch.load(model_path))
    
    # ---------------------- 5. 模型测试与评估 ----------------------
    # 初始化评估指标
    correct = 0  # 正确预测数
    total = 0    # 总样本数
    class_correct = list(0. for i in range(10))  # 各类别正确数
    class_total = list(0. for i in range(10))    # 各类别总样本数
    
    # 模型设为评估模式（关闭Dropout、BatchNorm等训练相关操作）
    net.eval()
    
    # 关闭梯度计算，减少内存消耗
    with torch.no_grad():
        for data in test_loader:
            images, labels = data  # 解包数据
            outputs = net(images)  # 前向传播
            _, predicted = torch.max(outputs, 1)  # 获取最大概率类别
            
            total += labels.size(0)  # 累计总样本数
            correct += (predicted == labels).sum().item()  # 累计正确数
            
            # 按类别统计准确率
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
    
    # ---------------------- 6. 计算并输出整体准确率 ----------------------
    print(f'模型在测试集上的整体准确率: {100 * correct / total:.2f}%')
    
    # ---------------------- 7. 输出各类别准确率 ----------------------
    for i in range(10):
        print(f'{classes[i]} 类别准确率: {100 * class_correct[i] / class_total[i]:.2f}%')
    
    # ---------------------- 8. 可视化错误案例（选做）----------------------
    visualize_misclassifications(net, test_loader, classes, num_samples=6)


def visualize_misclassifications(net, dataloader, classes, num_samples=6):
    """可视化模型误分类的样本"""
    net.eval()
    misclassified_images = []
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            
            # 找出误分类样本
            mis_idx = (predicted != labels)
            if mis_idx.sum() == 0:
                continue  # 若无误分类，跳过此批次
            
            # 收集误分类样本
            mis_images = images[mis_idx]
            mis_true = labels[mis_idx]
            mis_pred = predicted[mis_idx]
            
            misclassified_images.extend(mis_images[:num_samples])
            true_labels.extend(mis_true[:num_samples])
            pred_labels.extend(mis_pred[:num_samples])
            
            # 收集足够样本后停止
            if len(misclassified_images) >= num_samples:
                break
    
    # 绘制误分类样本
    plt.figure(figsize=(12, num_samples*2))
    for i in range(min(num_samples, len(misclassified_images))):
        plt.subplot(1, num_samples, i+1)
        img = misclassified_images[i].numpy()
        img = np.transpose(img, (1, 2, 0))  # 转换为HWC格式
        img = img * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))  # 反标准化
        img = np.clip(img, 0, 1)  # 确保像素值在[0,1]范围内
        
        plt.imshow(img)
        plt.title(f"真实: {classes[true_labels[i]]}\n预测: {classes[pred_labels[i]]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("misclassified_samples.png")
    plt.show()


if __name__ == '__main__':
    main()
