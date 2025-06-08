import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from matplotlib import pyplot as plt
import torch.nn as nn
import numpy as np
import random

# 优化后的LeNet5模型（使用ReLU和MaxPool）
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu = nn.ReLU()  # 替换Sigmoid为ReLU
        self.pool1 = nn.MaxPool2d(2, 2)  # 最大池化
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))  # 卷积+ReLU
        x = self.pool1(x)  # 池化
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))  # 在全连接层添加ReLU（可选优化）
        x = self.fc2(x)
        return x

# 数据预处理（保持不变）
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据集
train_dataset = datasets.MNIST('./MNIST', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./MNIST', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
