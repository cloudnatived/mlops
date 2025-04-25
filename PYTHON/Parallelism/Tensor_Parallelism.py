import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(100, 200)  # 第一个线性层
        self.fc2 = nn.Linear(200, 10)   # 第二个线性层

    def forward(self, x):
        # 模拟张量并行：拆分输入并在不同设备上计算
        # 将输入拆分成两部分，分别送到不同的GPU上
        x1, x2 = x.chunk(2, dim=1)  # 将输入拆成两部分，假设输入是[64, 100]
        x1 = x1.to(device1)  # 第一部分数据送到GPU1
        x2 = x2.to(device2)  # 第二部分数据送到GPU2
        
        # 在GPU1上计算第一部分
        out1 = self.fc1(x1)  # 在GPU1上计算
        out1 = out1.to(device2)  # 将输出转移到GPU2
        
        # 在GPU2上计算第二部分
        out2 = self.fc2(x2)  # 在GPU2上计算
        
        # 合并两个输出结果
        out = out1 + out2  # 假设这里是两个部分的合并（可以是加法、拼接等）
        return out

# 假设我们有两个GPU
device1 = torch.device("cuda:0")  # GPU1
device2 = torch.device("cuda:1")  # GPU2

model = Model().to(device1)  # 将模型的第一部分（fc1）放到GPU1

# 模拟输入数据
input_data = torch.randn(64, 100).to(device1)  # 假设输入数据是64个样本，每个样本100维

# 计算前向传播
output_data = model(input_data)

# 最终输出在GPU2上
print(output_data)
