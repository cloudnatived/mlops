



# tensorboard的基本使用及案例
https://blog.csdn.net/qq_52964132/article/details/145412008

```

pip3 install tensorboard
pip3 install torch torchvision

tensorboard --logdir=runs --host=10.0.10.249


TensorBoard 的可视化功能
    标量可视化：使用 add_scalar 方法记录标量数据，如损失值或准确率。
    图像可视化：使用 add_image 方法记录图像数据。
    直方图可视化：使用 add_histogram 方法记录直方图数据。
    图形可视化：使用 add_graph 方法记录模型的计算图。
    嵌入可视化：使用 add_embedding 方法记录嵌入数据。

```

```

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
 
# 设置 TensorBoard
writer = SummaryWriter('runs/embedding')
 
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
 
# 加载数据集
trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
 
# 定义一个简单的模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入是 28x28 的图像
        self.fc2 = nn.Linear(128, 10)  # 输出是 10 类
 
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 将图像展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
model = SimpleNet()
 
# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
 
# 训练模型
for epoch in range(10):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
 
        # 每隔一定步数记录一次嵌入数据
        if batch_idx % 100 == 0:
            # 获取嵌入层的输出
            embedding = model.fc1(data.view(-1, 28 * 28)).detach().numpy()
            # 获取标签
            labels = target.numpy()
            # 记录嵌入数据
            writer.add_embedding(
                mat=embedding,
                metadata=labels,
                label_img=data.unsqueeze(1),
                global_step=epoch * len(trainloader) + batch_idx
            )
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
 
writer.close()

```
