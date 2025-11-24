# From https://zhuanlan.zhihu.com/p/657602028
# 导入PyTorch库
# Accuracy on the test set: 98.69% 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 定义LeNet-5架构的神经网络类
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)                            # 第一卷积层：输入1通道（灰度图像），输出6通道，卷积核大小为5x5
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                     # 第一池化层：最大池化，池化窗口大小为2x2，步幅为2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)                           # 第二卷积层：输入6通道，输出16通道，卷积核大小为5x5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                     # 第二池化层：最大池化，池化窗口大小为2x2，步幅为2
        self.fc1 = nn.Linear(16 * 4 * 4, 120)                                  # 第一个全连接层：输入维度是16*4*4，输出维度是120
        self.fc2 = nn.Linear(120, 84)                                          # 第二个全连接层：输入维度是120，输出维度是84
        self.fc3 = nn.Linear(84, 10)                                           # 第三个全连接层：输入维度是84，输出维度是10，对应10个类别

    def forward(self, x):
        # 前向传播函数定义网络的数据流向
        x = self.pool1(torch.relu(self.conv1(x)))                              # 第一层卷积 + 激活 + 池化
        x = self.pool2(torch.relu(self.conv2(x)))                              # 第二层卷积 + 激活 + 池化
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))                                            # 全连接层 + 激活
        x = self.fc3(x)                                                        # 输出层
        return x

# 定义数据变换和加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 训练数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 测试数据集
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化LeNet-5模型以及定义损失函数和优化器
net = LeNet5()
criterion = nn.CrossEntropyLoss()                                     # 交叉熵损失函数，用于分类问题，分类任务使用交叉熵损失
optimizer = optim.Adam(net.parameters(), lr=0.001)                    # 设置优化器，设置Adam优化器，学习率为0.001

# 训练循环
for epoch in range(5):                                                # 可以根据需要调整训练的轮数
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()                                         # 清零梯度，清除之前计算的梯度
        outputs = net(inputs)                                         # 前向传播                   
        loss = criterion(outputs, labels)                             # 计算损失
        loss.backward()                                               # 反向传播，计算梯度
        optimizer.step()                                              # 更新权重

        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
    torch.save(net.state_dict(), 'LeNet5_0.pth')                      # 保存模型

print("Finished Training")

# 测试模型
correct = 0                                                           # 初始化正确预测的数
total = 0                                                             # 初始化总的样本数
with torch.no_grad():                                                 # 不需计算梯度，这样可以节省内存和计算资源
    for data in test_loader:                                          # 遍历数据加载器中的每一个 batch
        inputs, labels = data                                         # 将输入数据和标签转移到指定的设备
        outputs = net(inputs)                                         # 用当前的模型进行前向传播，得到预测结果
        _, predicted = torch.max(outputs.data, 1)                     # 找到最大概率的类别，对每个输出的预测结果，找到最大值对应的索引，返回预测的类标签
        total += labels.size(0)                                       # 更新总样本数， labels.size(0) 是当前 batch 的样本数
        correct += (predicted == labels).sum().item()                 # 统计预测正确的样本

accuracy = 100 * correct / total                                      # 计算准确率
print(f"Accuracy on the test set: {accuracy}%")                       # 打印计算得到的准确率
