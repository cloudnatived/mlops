import torch
from torch import nn
# 神经网络的实现代码
# 实现任何的深度学习模型，最重要的是两步骤。
# 定义神经网络Network，继承为nn.Module类

class Network(nn.Module):    # 步骤1:弄懂这个模型中“有什么"? 模型中有哪些结构? 神经网络中的神经元的数量是固定的，所以init不用传入参数
    def __init__(self): 
        super().__init__()                      # 调用了父类的初始化函数
        self.layer1 = nn.Linear( 28 * 28, 256)  # layer1是输入层与隐藏层之间的线性层
        self.layer2 = nn.Linear(256, 10)        # layer2是隐藏层与输出层之间的线性层
        
    # 步骤2:弄懂这个模型"怎么算"? 这个模型如何进行推理，如何进行前向传播计算?    
    # 在forward函数中，定义该神经网络的推理计算过程
    # 函数输入张量x，x的尺寸是nx28x28，其中n表示了n张图片
    # 在前向传播，forward函数中，输入为图像x
    def forward(self, x):    # 步骤2:弄懂这个模型"怎么算"? 这个模型如何进行推理，如何进行前向传播计算? 在forward函数中，定义该神经网络的推理计算过程
        x = x.view(-1, 28 * 28)     # 使用view函数，将向量x展平，尺寸变为nx784
        x = self.layer1(x)          # 将x输入至layer1
        x = torch.relu(x)           # 使用relu激活
        x = self.layer2(x)          # 输入至layer2计算结果
        return x                    # 返回结果

def print_parameters(model):                #手动的遍历模型中的各个结构，并计算可以训练的参数
    cnt = 0
    for name, layer in model.named_children():     #遍历每一层
        print(f"layer({name}) parameters:") # 打印层的名称和该层中包含的可训练参数
        for p in layer.parameters():
            print(f'\t {p.shape} has {p.numel()} parameters')
            cnt += p.numel()                      #将参数数量累加至cnt
    print('The model has %d trainable parameters\n' % (cnt)) #最后打印模型总参数数量

def print_forward(model, x):            #打印输入张量x经过每一层时的维度变化情况
    print(f"x: {x.shape}")              # x从一个5*28*28的输入张量
    x = x.view(-1, 28 * 28)             # 经过view函数，变成了一个5*784的张量
    print(f"after view: {x.shape}")
    x = model.layer1(x)                 # 经过第1个线性层，得到5*256的张量
    print(f"after layer1: {x.shape}")
    x = torch.relu(x)                   # 经过relu函数，没有变化
    print(f"after relu: {x.shape}")
    x = model.layer2(x)                 # 经过第2个线性层，得到一个5*10的结果
    print(f"after layer2: {x.shape}")

if __name__ == '__main__':
    model = Network()                   # 定义一个Network模型
    print(model)                        # 将其打印，观察打印结果可以了解模型的结构
    print("")

    print_parameters(model)             #将模型的参数打印出来
    x = torch.zeros([5, 28, 28])        # 定义一个5x28x28的张量x，表示了5个28x28的图像
    print_forward(model, x)             # 打印输入张量x经过每一层维度的变化情况
