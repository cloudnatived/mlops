# From https://blog.csdn.net/2403_83575015/article/details/143755629
# chatgpt优化
import torch                                         # 导入PyTorch库
from torchvision import datasets, transforms         # 从torchvision导入数据集和数据转换工具
from torch.utils.data import DataLoader              # 导入DataLoader，用于加载数据
import torch.optim as optim                          # 导入优化器工具
from matplotlib import pyplot as plt                 # 导入matplotlib，用于绘图
import torch.nn as nn                                # 导入PyTorch的神经网络模块
import numpy as np                                   # 导入NumPy，用于数组操作
import random                                        # 导入random，用于随机操作


# 定义LeNet5网络模型,优化后的LeNet5模型（使用ReLU和MaxPool）
class LeNet5(nn.Module):                             # 继承自nn.Module类，定义LeNet5网络结构
    def __init__(self):                              # 初始化网络结构
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)              # 第一层卷积：输入1个通道，输出6个通道，卷积核大小5x5
        #self.Sigmoid = nn.Sigmoid()                 # Sigmoid激活函数
        self.relu = nn.ReLU()                        # 替换Sigmoid为ReLU
        self.pool1 = nn.MaxPool2d(2, 2)              # 第二层池化：2x2大小的平均池化
        self.conv2 = nn.Conv2d(6, 16, 5)             # 第三层卷积：输入6个通道，输出16个通道，卷积核大小5x5
        self.pool2 = nn.MaxPool2d(2, 2)              # 第四层池化：2x2大小的平均池化     
        self.conv3 = nn.Conv2d(16, 120, 5)           # 第五层卷积：输入16个通道，输出120个通道，卷积核大小5x5
        self.flatten = nn.Flatten()                  # 展平层：将卷积层的输出（多维数据）展平为一维，便于全连接层输入
        self.fc1 = nn.Linear(120, 84)                # 第六层全连接：输入120个特征，输出84个特征
        self.fc2 = nn.Linear(84, 10)                 # 输出层：最终的全连接层，输出10个类别（MNIST有10个数字分类）

    def forward(self, x):
        #x = self.Sigmoid(self.c1(x))                # 第一层卷积后加激活函数
        x = self.relu(self.conv1(x))                 # 卷积+ReLU，第一层卷积后加激活函数
        x = self.pool1(x)                            # 第一层池化
        x = self.relu(self.conv2(x))                 # 第二层卷积后加激活函数
        x = self.pool2(x)                            # 第二层池化
        x = self.conv3(x)                            # 第三层卷积
        x = self.flatten(x)                          # 展平操作
        x = self.relu(self.fc1(x))                   # 在全连接层添加ReLU（可选优化）
        x = self.fc2(x)                              # 输出层   
        return x

# 数据预处理，包括调整图片大小、转换为Tensor并进行归一化
transform = transforms.Compose([               
    transforms.Resize((32, 32)),                     # 将图片大小调整为32x32像素
    transforms.ToTensor(),                           # 将PIL图片或NumPy ndarray转换为Tensor
    transforms.Normalize((0.5,), (0.5,))             # 归一化，将图像像素值从[0, 1]区间调整到[-1, 1]
])

# 加载数据集
#train_dataset = datasets.MNIST('./MNIST', train=True, download=True, transform=transform)
train_dataset = datasets.MNIST('/Data/DEMO/MODEL/MNIST', train=True, download=True, transform=transform)
#test_dataset = datasets.MNIST('./MNIST', train=False, download=True, transform=transform)
test_dataset = datasets.MNIST('/Data/DEMO/MODEL/MNIST', train=False, download=True, transform=transform)

# 创建数据加载器，使用DataLoader将数据分批处理
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)   # 训练集，批次大小64，数据打乱
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)    # 测试集，批次大小64，不打乱数据

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)  # 添加学习率衰减

# 训练历史记录初始化（移至循环前）
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

# 训练与测试循环
epochs = 20
for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        _, pred = torch.max(output, 1)
        total_train += target.size(0)
        correct_train += (pred == target).sum().item()

    # 计算训练指标
    train_loss = running_train_loss / len(train_loader)
    train_acc = 100.0 * correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # 测试阶段（优化可视化逻辑，仅采样当前批次数据）
    model.eval()
    running_test_loss = 0.0
    correct_test = 0
    total_test = 0
    visualized = False  # 控制仅可视化一次

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_test_loss += loss.item()

            _, pred = torch.max(output, 1)
            total_test += target.size(0)
            correct_test += (pred == target).sum().item()

            # 仅在第一个批次中随机选取10张图像可视化
            if not visualized and i == 0:
                num_vis = 10
                indices = random.sample(range(len(data)), num_vis)
                vis_images = data[indices].cpu().numpy()
                vis_true = target[indices].cpu().numpy()
                vis_pred = pred[indices].cpu().numpy()
                visualized = True

    # 计算测试指标
    test_loss = running_test_loss / len(test_loader)
    test_acc = 100.0 * correct_test / total_test
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    # 打印结果
    print(f'Epoch {epoch+1}/{epochs}')
    print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%\n')

    # 调用学习率衰减
    scheduler.step()

# 可视化预测结果（仅最后一次测试的采样数据）
def plot_predictions(images, true_labels, pred_labels):
    plt.figure(figsize=(12, 2))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i+1)
        ax.axis('off')
        img = np.squeeze(images[i])
        ax.imshow(img, cmap='gray')
        ax.set_title(f'T:{true_labels[i]}\nP:{pred_labels[i]}', fontsize=8)
    plt.tight_layout()
    plt.show()

plot_predictions(vis_images, vis_true, vis_pred)

# 绘制损失和准确率曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(test_accuracies, label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()
