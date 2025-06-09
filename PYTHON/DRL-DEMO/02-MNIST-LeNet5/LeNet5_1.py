# From AI 
# Test set: Average loss: 0.0370, Accuracy: 9897/10000 (99%)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 定义 LeNet-5 网络结构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()                                                          
        self.conv1 = nn.Conv2d(1, 6, 5)                  # 第一层卷积层：输入1通道，输出6通道，卷积核5x5，padding=same
        self.conv2 = nn.Conv2d(6, 16, 5)                 # 第二层卷积层：输入6通道，输出16通道，卷积核5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)            # 第三层卷积层：输入16通道，输出120通道，卷积核5x5
        self.fc2 = nn.Linear(120, 84)                    # 全连接层：输入120，输出84
        self.fc3 = nn.Linear(84, 10)                     # 全连接层：输入84，输出10（对应10个类别）

    def forward(self, x):                                # 前向传播函数，定义数据流向
        x = F.relu(self.conv1(x))                        # 第一层卷积 + 激活 + 池化
        x = F.max_pool2d(x, 2, 2)                        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((32, 32)),                          # 将图像调整为32x32，LeNet 输入是 32x32，符合LeNet-5输入要求
    transforms.ToTensor(),                                # 转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,))            # MNIST数据集的均值和标准差
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 设置设备、模型和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LeNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()                            # 5.将梯度清零
        output = model(data)                             # 1.计算神经网络的前向传播结果
        loss = F.cross_entropy(output, target)           # 2.计算output和标签label之间的损失loss  
        loss.backward()                                  # 3.使用backward计算梯度
        optimizer.step()                                 # 4.使用optimizer.step更新参数
        if batch_idx % 100 == 0:                         # 每迭代100个小批量，就打印一次模型的损失，观察训练的过程
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

# 训练和测试循环
for epoch in range(1, 11):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# 保存模型
#torch.save(model.state_dict(), 'lenet5_mnist.pth')
torch.save(model.state_dict(), 'LeNet5_1.pth')
