import torch
import torch.nn as nn

class Part1(nn.Module):
    def __init__(self):
        super(Part1, self).__init__()
        self.fc1 = nn.Linear(100, 200)

    def forward(self, x):
        return torch.relu(self.fc1(x))

class Part2(nn.Module):
    def __init__(self):
        super(Part2, self).__init__()
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        return self.fc2(x)

device1 = torch.device("cuda:0")
device2 = torch.device("cuda:1")

model_part1 = Part1().to(device1)
model_part2 = Part2().to(device2)

# 假设输入数据
input_data = torch.randn(64, 100).to(device1)

# 在GPU 1上计算第一部分
h1 = model_part1(input_data)

# 将数据传输到GPU 2并计算第二部分
h1 = h1.to(device2)
output = model_part2(h1)

