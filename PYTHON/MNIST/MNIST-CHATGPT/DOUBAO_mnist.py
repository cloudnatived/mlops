# FROM AI DOUBAO
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")      # 检查CUDA配置

# 数据预处理：MNIST为28x28灰度图，无需Resize
transform = transforms.Compose([
    transforms.RandomRotation(10),                                         # 随机旋转±10度
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))                             # MNIST预定义均值/标准差
])

# 加载数据集
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# 定义模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),                       # 第一层卷积块，输入1通道，输出32通道，3x3卷积核，padding=1保持尺寸28x28
            nn.BatchNorm2d(32),                                               # 批量归一化，加速收敛
            nn.ReLU(),                                                        # 激活函数
            nn.MaxPool2d(2),                                                  # 池化层，尺寸减半为14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),                      # 第二层卷积块，输入32通道，输出64通道，尺寸保持14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)                                                   # 池化后尺寸7x7
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),                                       # 全连接层，展平后维度3136→128
            nn.Dropout(0.5),                                                  # 随机丢弃50%神经元，防止过拟合
            nn.ReLU(),
            nn.Linear(128, 10)                                                # 输出10类（0-9）
        )

    def forward(self, x):
        x = self.conv(x)                                                      # 卷积+池化处理
        x = x.view(x.size(0), -1)                                             # 展平张量，-1自动计算剩余维度（3136），动态展平，兼容不同批量大小
        return self.fc(x)                                                     # 全连接层分类

# 初始化模型、损失函数和优化器
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()                                             # 交叉熵损失函数（适用于多分类）
optimizer = optim.Adam(model.parameters(), lr=1e-3)                           # Adam优化器，初始学习率0.001
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)      # 学习率衰减，每5个epoch学习率减半

# 训练函数
def train_model(model, train_loader, criterion, optimizer, epoch, device):
    model.train()                                                             # 开启训练模式（Dropout和BatchNorm生效）
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()                                                 # 5.将梯度清零（避免累加）
        outputs = model(data)                                                 # 1.计算神经网络的前向传播结果
        loss = criterion(outputs, targets)                                    # 2.计算output和标签label之间的损失loss
        loss.backward()                                                       # 3.使用backward计算梯度
        optimizer.step()                                                      # 4.使用optimizer.step更新参数
        
        running_loss += loss.item()                                           # 统计损失和准确率
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        if batch_idx % 100 == 0:                                              # 每迭代100个小批量，就打印一次模型的损失，观察训练的过程
            print(f"Epoch [{epoch+1}/10], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")
    return running_loss / len(train_loader), 100.*correct/total

# 评估函数
def evaluate_model(model, test_loader, criterion, device):
    model.eval()                                                              # 开启评估模式（Dropout和BatchNorm不更新参数）
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():                                                     # 关闭梯度计算（节省内存和计算量）                                
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()                                       # 前向传播
            _, predicted = torch.max(outputs.data, 1)                         # 找到最大概率的类别
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return running_loss / len(test_loader), 100.*correct/total

# 训练主循环
num_epochs = 10                                                               # 增加训练轮数
train_losses, train_accs, test_losses, test_accs = [], [], [], []

for epoch in range(num_epochs):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, epoch, device)
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    
    scheduler.step()                                                          # 调整学习率，按epoch衰减学习率
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    print(f"\nEpoch {epoch+1}/{num_epochs} Complete")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%\n")

# 打印最终结果
print("Training Finished!")
print(f"Final Test Accuracy: {test_accs[-1]:.2f}%")

torch.save(model.state_dict(), "DOUBAO_mnist.pth")
print("模型已保存为 DOUBAO_mnist.pth")                                              # 保存模型参数（而非整个模型）
