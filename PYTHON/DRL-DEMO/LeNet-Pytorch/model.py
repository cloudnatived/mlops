import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):                       # 继承来自nn.Module的父类。
    """LeNet卷积神经网络模型，经典的图像分类架构"""
    def __init__(self):                       # 初始化网络
        """初始化网络层结构"""
        super(LeNet, self).__init__()         # 调用父类(nn.Module)的构造函数，super()继承父类的构造函数，多继承需用到super函数。
        self.conv1 = nn.Conv2d(3, 16, 5)      # 第一层：卷积层 + 激活函数。输入通道数：3（RGB彩色图像）。输出通道数：16个特征图。卷积核大小：5×5。
        self.pool1 = nn.MaxPool2d(2, 2)       # 第一层：池化层。池化窗口大小：2×2。步长：2（与窗口大小相同，无重叠）。
        self.conv2 = nn.Conv2d(16, 32, 5)     # 第二层：卷积层 + 激活函数。输入通道数：16（承接上一层输出）。输出通道数：32个特征图。卷积核大小：5×5。
        self.pool2 = nn.MaxPool2d(2, 2)       # 第二层：池化层。池化窗口大小：2×2。步长：2。
        self.fc1 = nn.Linear(32*5*5, 120)     # 第二层：卷积层 + 激活函数。输入维度：32个特征图×5×5（池化后尺寸）。输出维度：120个神经元。
        self.fc2 = nn.Linear(120, 84)         # 全连接层2：特征维度映射。# 120→84。
        self.fc3 = nn.Linear(84, 10)          # 输出层：分类器。84→10（10个类别，如MNIST数字分类）。

    def forward(self, x):
        """定义前向传播流程"""
        x = F.relu(self.conv1(x))    # 第一层卷积+激活。input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # 第一层池化。output(16, 14, 14)
        x = F.relu(self.conv2(x))    # 第二层卷积+激活。output(32, 10, 10)
        x = self.pool2(x)            # 第二层池化。output(32, 5, 5)。
        x = x.view(-1, 32*5*5)       # 展平特征图以便全连接。输出形状：(batch_size, 800)。output(32*5*5)
        x = F.relu(self.fc1(x))      # 全连接层1+激活。输出形状：(batch_size, 120)。output(120)
        x = F.relu(self.fc2(x))      # 全连接层2+激活。输出形状：(batch_size, 84)。output(84)
        x = self.fc3(x)              # 输出层（无激活函数，分类任务通常由后续Softmax处理）。输出形状：(batch_size, 10)。output(10)
        return x


