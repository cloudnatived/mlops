# From https://blog.csdn.net/2403_83575015/article/details/143755629
import torch  # 导入PyTorch库
from torchvision import datasets, transforms  # 从torchvision导入数据集和数据转换工具
from torch.utils.data import DataLoader  # 导入DataLoader，用于加载数据
import torch.optim as optim  # 导入优化器工具
from matplotlib import pyplot as plt  # 导入matplotlib，用于绘图
import torch.nn as nn  # 导入PyTorch的神经网络模块
import numpy as np  # 导入NumPy，用于数组操作
import random  # 导入random，用于随机操作
 
# 定义LeNet5网络模型
class LeNet5(nn.Module):  # 继承自nn.Module类，定义LeNet5网络结构
    def __init__(self):  # 初始化网络结构
        super(LeNet5, self).__init__()
 
        # 第一层卷积：输入1个通道，输出6个通道，卷积核大小5x5
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
 
        # Sigmoid激活函数
        self.Sigmoid = nn.Sigmoid()
 
        # 第二层池化：2x2大小的平均池化
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
 
        # 第三层卷积：输入6个通道，输出16个通道，卷积核大小5x5
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
 
        # 第四层池化：2x2大小的平均池化
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
 
        # 第五层卷积：输入16个通道，输出120个通道，卷积核大小5x5
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
 
        # 展平层：将卷积层的输出（多维数据）展平为一维，便于全连接层输入
        self.flat = nn.Flatten()
 
        # 第六层全连接：输入120个特征，输出84个特征
        self.f6 = nn.Linear(120, 84)
 
        # 输出层：最终的全连接层，输出10个类别（MNIST有10个数字分类）
        self.output = nn.Linear(84, 10)
 
    # 前向传播函数，定义数据流向
    def forward(self, x):
        # 第一层卷积后加激活函数
        x = self.Sigmoid(self.c1(x))
        # 第一层池化
        x = self.s2(x)
        # 第二层卷积后加激活函数
        x = self.Sigmoid(self.c3(x))
        # 第二层池化
        x = self.s4(x)
        # 第三层卷积
        x = self.c5(x)
        # 展平操作
        x = self.flat(x)
        # 全连接层
        x = self.f6(x)
        # 输出层
        x = self.output(x)
        return x
 
 
# 数据预处理，包括调整图片大小、转换为Tensor并进行归一化
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 将图片大小调整为32x32像素
    transforms.ToTensor(),  # 将PIL图片或NumPy ndarray转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化，将图像像素值从[0, 1]区间调整到[-1, 1]
])
 
# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./MNIST', train=True, download=True, transform=transform)  # 加载训练数据集
test_dataset = datasets.MNIST(root='./MNIST', train=False, download=True, transform=transform)  # 加载测试数据集
 
# 创建数据加载器，使用DataLoader将数据分批处理
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 训练集，批次大小64，数据打乱
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # 测试集，批次大小64，不打乱数据
 
 
# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0  # 初始化损失
    correct = 0  # 初始化正确分类的数量
    total = 0  # 初始化总样本数
 
    # 遍历训练数据集
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)  # 将数据和标签传输到指定的计算设备（GPU/CPU）
 
        optimizer.zero_grad()  # 清除之前的梯度
        output = model(data)  # 前向传播，计算输出
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数
 
        running_loss += loss.item()  # 累计损失
        _, predicted = torch.max(output, 1)  # 获取输出中概率最大的类索引作为预测
        total += target.size(0)  # 样本总数
        correct += (predicted == target).sum().item()  # 统计正确分类的数量
 
    accuracy = 100.0 * correct / total  # 计算训练准确率
    avg_loss = running_loss / len(train_loader)  # 计算平均损失
    return avg_loss, accuracy
 
 
# 测试函数（改进版）
def test(model, test_loader, criterion, device, num_images_to_display=10):
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0  # 初始化损失
    correct = 0  # 初始化正确分类的数量
    total = 0  # 初始化总样本数
 
    # 用于存储测试集的图像和标签
    test_images = []
    test_labels = []
    predicted_labels = []
 
    # 测试模式下不需要计算梯度
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # 将数据传输到设备上
            output = model(data)  # 前向传播，得到输出
            loss = criterion(output, target)  # 计算损失
            running_loss += loss.item()  # 累计损失
 
            _, predicted = torch.max(output, 1)  # 获取预测标签
            #import torch
            # output = torch.tensor([
            #     [0.1, 0.2, 0.5, 0.4],
            #     [0.7, 0.6, 0.8, 0.3],
            #     [0.9, 0.3, 0.2, 0.6]
            # ])
            # max_values, max_indices = torch.max(output, 1)
            # print("Max values:", max_values) 值
            # print("Max indices:", max_indices) 索引
            #Max values: tensor([0.5, 0.8, 0.9])
            # Max indices: tensor([2, 2, 0])
 
            total += target.size(0)  # 总样本数
            correct += (predicted == target).sum().item()  # 计算正确预测的数量
 
            # total += target.size(0)：累加当前批次的样本数，target.size(0)
            # 返回批次的大小（即每批次中的样本数量）。
            # correct += (predicted == target).sum().item()：计算模型预测正确的数量。predicted == target
            # 返回一个布尔值的Tensor，表示预测是否与真实标签一致，.sum()
            # 汇总正确的数量，.item()
            # 将结果从Tensor转换为标量。
 
            # 将当前批次的图片、标签和预测结果存储起来
            test_images.extend(data.cpu().numpy())  # 存储图像数据
            test_labels.extend(target.cpu().numpy())  # 存储真实标签
            predicted_labels.extend(predicted.cpu().numpy())  # 存储预测标签
 
            # data.cpu().numpy()将数据从GPU转移到CPU，然后转换为NumPy数组，存储在test_images列表中。
            # 同样，真实标签target和预测标签predicted也分别转为NumPy数组，存储在test_labels和predicted_labels中。
 
    accuracy = 100.0 * correct / total  # 计算测试集准确率
    avg_loss = running_loss / len(test_loader)  # 计算测试集损失
 
    # 随机选取需要展示的图像
    indices = random.sample(range(len(test_images)), num_images_to_display)
    #random.sample 从所有测试图像中随机选择 num_images_to_display 张图片，生成一个包含随机索引的列表
 
    # 根据随机选出的索引获取图片、真实标签和预测标签
    selected_images = [test_images[i] for i in indices]
    selected_true_labels = [test_labels[i] for i in indices]
    selected_predicted_labels = [predicted_labels[i] for i in indices]
 
    # 调用可视化函数，展示预测结果
    plot_predictions(selected_images, selected_true_labels, selected_predicted_labels)
 
    return avg_loss, accuracy
 
 
# 可视化函数，展示预测图片和标签
def plot_predictions(images, true_labels, predicted_labels):
    num_images = len(images)  # 获取展示的图片数量
    num_columns = 5  # 每行显示5张图片
    num_rows = num_images // num_columns + (num_images % num_columns != 0)  # 计算所需的行数
 
    # 创建绘图窗口，调整大小
    plt.figure(figsize=(12, num_rows * 2))
 
    # 遍历每张图片进行展示
    for i in range(num_images):
        ax = plt.subplot(num_rows, num_columns, i + 1)  # 创建子图
        ax.axis('off')  # 不显示坐标轴
        img = np.squeeze(images[i])  # 将图像从(1, 32, 32)转换为(32, 32)
        ax.imshow(img, cmap='gray')  # 使用灰度图显示图片
        # 显示真实标签和预测标签
        ax.set_title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}")
 
    plt.tight_layout()  # 自动调整子图之间的间距
    plt.show()  # 展示图片
 
 
# 设置设备为GPU（如果可用），否则为CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# 初始化LeNet5模型
model = LeNet5().to(device)
 
# 定义损失函数（交叉熵损失）和优化器（Adam）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
# 训练并测试模型，设置训练轮数
epochs = 20
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')  # 输出当前训练轮次
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)  # 训练过程
    test_loss, test_acc = test(model, test_loader, criterion, device, num_images_to_display=10)  # 测试过程
 
    # 打印每轮的训练和测试结果
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
 
# 记录训练过程中的损失和准确率
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []
 
for epoch in range(epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)  # 训练
    test_loss, test_acc = test(model, test_loader, criterion, device, num_images_to_display=10)  # 测试
 
    # 保存损失和准确率数据
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
 
# 绘制训练损失和测试损失图
plt.figure(figsize=(12, 5))
 
# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(epochs), train_losses, label='Train Loss')  # 绘制训练损失
plt.plot(range(epochs), test_losses, label='Test Loss')  # 绘制测试损失
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
 
# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(epochs), train_accuracies, label='Train Accuracy')  # 绘制训练准确率
plt.plot(range(epochs), test_accuracies, label='Test Accuracy')  # 绘制测试准确率
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
 
plt.show()  # 显示损失和准确率图
