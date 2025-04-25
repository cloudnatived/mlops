import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel

# 创建一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 假设我们有两个GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN()

# 使用DataParallel包装模型，进行数据并行训练
model = DataParallel(model)  # 自动分配到多个GPU
model.to(device)

# 假设数据
input_data = torch.randn(64, 100).to(device)  # 一个批次的数据
target = torch.randn(64, 10).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(10):
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

