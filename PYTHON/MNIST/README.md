

# MNIST 数据集
  

## /MNIST-XHH   -- 小黑黑 mnist_network
### 三层神经网络训练手写数字识别

```
# ubuntu 22.04.5 下的环境准备
cp /etc/apt/sources.list /etc/apt/sources.list.original;
cat > /etc/apt/sources.list <<EOF
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-proposed main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-proposed main restricted universe multiverse
EOF

apt update -y;
apt list --upgradable;
apt upgrade -y;

apt install -y python3-pip python3 python3-netaddr wget git;
apt install -y python3-dev;
pip install --upgrade pip;

# 安装python3和pip3
apt install python3-pip;
pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple;
mkdir -p /root/.config/pip;
cat > /root/.config/pip/pip.conf <<EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
EOF

python3 -m pip install --upgrade pip;

# 程序所在目录，下载这5个文件
https://github.com/cloudnatived/mlops/tree/main/PYTHON/MNIST/MNIST-XHH
# 下载这5个文件
requirements.txt
model.py
download_data.py
train.py
test.py

# 安装项目所需的python库
pip3 install -r requirements.txt;

# 下载数据集
python3 download_data.py

# 训练模型，讲模型保存成mnist.pth
python3 train.py

# 用测试数据集测试训练好的模型
python3 test.py

准确率：
test accuracy = 9801 / 10000 = 0.980


```




```

# requirements.txt
###########################################################
matplotlib              # Python 的绘图库，它能让使用者很轻松地将数据图形化，可以用来绘制各种静态，动态，交互式的图表，并且提供多样化的输出格式
numpy                   # 支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库
Pillow==11.2.1          # Python3 最常用的图像处理库
tensorflow==2.19.0      # 开源机器学习框架，致力于数据流图的自动微分和深度神经网络计算
torch==2.7.0            # 开源的机器学习库，主要用于进行计算机视觉（CV）、自然语言处理（NLP）、语音识别等领域的研究和开发 
torchvision==0.22.0     # 主要处理图像数据，包含一些常用的数据集、模型、转换函数等
###########################################################

# model.py 定义模型
###########################################################
import torch
from torch import nn
# 神经网络的实现代码
# 实现任何的深度学习模型，最重要的两步骤
# 定义神经网络Network，继承为nn.Module类

class Network(nn.Module):
    # 步骤1:弄懂这个模型中“有什么"? 模型中有哪些结构?
    # 神经网络中的神经元的数量是固定的，所以init不用传入参数
    def __init__(self): 
        super().__init__() # 调用了父类的初始化函数
        # layer1是输入层与隐藏层之间的线性层
        self.layer1 = nn.Linear(784, 256)
        # layer2是隐藏层与输出层之间的线性层
        self.layer2 = nn.Linear(256, 10)
        
    # 步骤2:弄懂这个模型"怎么算"? 这个模型如何进行推理，如何进行前向传播计算?    
    # 在forward函数中，定义该神经网络的推理计算过程
    # 函数输入张量x，x的尺寸是nx28x28，其中n表示了n张图片
    # 在前向传播，forward函数中，输入为图像x
    def forward(self, x):
        x = x.view(-1, 28 * 28) # 使用view函数，将向量x展平，尺寸变为nx784
        x = self.layer1(x)      # 将x输入至layer1
        x = torch.relu(x)       # 使用relu激活
        x = self.layer2(x)      # 输入至layer2计算结果
        return x # 返回结果

#手动的遍历模型中的各个结构，并计算可以训练的参数
def print_parameters(model):
    cnt = 0
    for name, layer in model.named_children(): #遍历每一层
        # 打印层的名称和该层中包含的可训练参数
        print(f"layer({name}) parameters:")
        for p in layer.parameters():
            print(f'\t {p.shape} has {p.numel()} parameters')
            cnt += p.numel() #将参数数量累加至cnt
    #最后打印模型总参数数量
    print('The model has %d trainable parameters\n' % (cnt))

#打印输入张量x经过每一层时的维度变化情况
def print_forward(model, x):
    print(f"x: {x.shape}") # x从一个5*28*28的输入张量
    x = x.view(-1, 28 * 28) # 经过view函数，变成了一个5*784的张量
    print(f"after view: {x.shape}")
    x = model.layer1(x) # 经过第1个线性层，得到5*256的张量
    print(f"after layer1: {x.shape}")
    x = torch.relu(x) # 经过relu函数，没有变化
    print(f"after relu: {x.shape}")
    x = model.layer2(x) # 经过第2个线性层，得到一个5*10的结果
    print(f"after layer2: {x.shape}")

if __name__ == '__main__':
    model = Network() # 定义一个Network模型
    print(model) # 将其打印，观察打印结果可以了解模型的结构
    print("")

    print_parameters(model) #将模型的参数打印出来
    # 定义一个5x28x28的张量x，表示了5个28x28的图像
    # 打印输入张量x经过每一层维度的变化情况
    x = torch.zeros([5, 28, 28])
    print_forward(model, x)
###########################################################

# download_data.py 下载数据集
###########################################################
# 这个程序的功能会先将MNIST数据下载下来，然后再保存为.png的格式。

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# 注意，这里得到的train_data和test_data已经直接可以用于训练了！
# 不一定要继续后面的保存图像。
train_data = MNIST(root='./data', train=True, download=True, transform=ToTensor())
test_data = MNIST(root='./data', train=False, download=True, transform=ToTensor())

# 继续将手写数字图像“保存”下来
# 输出两个文件夹，train和test，分别保存训练和测试数据。
# train和test文件夹中，分别有0、1、2、3、4、5、6、7、8、9；
# 这10个子文件夹保存每种数字的数据

from torchvision.transforms import ToPILImage

train_data = [(ToPILImage()(img), label) for img, label in train_data]
test_data = [(ToPILImage()(img), label) for img, label in test_data]

import os
import secrets
def save_images(dataset, folder_name):
    root_dir = os.path.join('./mnist_images', folder_name)
    os.makedirs(root_dir, exist_ok=True)
    for i in range(len(dataset)):
        img, label = dataset[i]
        label_dir = os.path.join(root_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        random_filename = secrets.token_hex(8) + '.png'
        img.save(os.path.join(label_dir, random_filename))

save_images(train_data, 'train')
save_images(test_data, 'test')
###########################################################


# train.py 模型训练
###########################################################
import torch
from torch import nn
from torch import optim

from model import Network

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # 图像的预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图
        transforms.ToTensor()  # 转换为张量
    ])

    # 读入并构造数据集
    train_dataset = datasets.ImageFolder(root='./mnist_images/train', transform=transform)
    print("train_dataset length: ", len(train_dataset))

    # 小批量的数据读入
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print("train_loader length: ", len(train_loader))

    model = Network()  # 模型本身，它就是我们设计的神经网络
    optimizer = optim.Adam(model.parameters())  # 优化模型中的参数
    criterion = nn.CrossEntropyLoss()  # 分类问题，使用交叉熵损失误差

    # 进入模型的迭代循环
    for epoch in range(10):  # 外层循环，代表了整个训练数据集的遍历次数
        # 整个训练集要循环多少轮，是10次、20次或者100次都是可能的，

        # 内存循环使用train_loader，进行小批量的数据读取
        for batch_idx, (data, label) in enumerate(train_loader):
            # 内层每循环一次，就会进行一次梯度下降算法
            # 包括了5个步骤:
            output = model(data) # 1.计算神经网络的前向传播结果
            loss = criterion(output, label) # 2.计算output和标签label之间的损失loss
            loss.backward()  # 3.使用backward计算梯度
            optimizer.step()  # 4.使用optimizer.step更新参数
            optimizer.zero_grad()  # 5.将梯度清零
            # 这5个步骤，是使用pytorch框架训练模型的定式，初学的时候，先记住就可以了

            # 每迭代100个小批量，就打印一次模型的损失，观察训练的过程
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/10 "
                      f"| Batch {batch_idx}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'mnist.pth') # 保存模型
###########################################################


# test.py 模型测试
###########################################################
from model import Network
from torchvision import transforms
from torchvision import datasets
import torch

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    # 读取测试数据集
    test_dataset = datasets.ImageFolder(root='./mnist_images/test', transform=transform)
    print("test_dataset length: ", len(test_dataset))

    model = Network()  # 定义神经网络模型
    model.load_state_dict(torch.load('mnist.pth')) # 加载刚刚训练好的模型文件

    right = 0 # 保存正确识别的数量
    for i, (x, y) in enumerate(test_dataset):
        output = model(x)  # 将其中的数据x输入到模型
        predict = output.argmax(1).item() # 选择概率最大标签的作为预测结果
        # 对比预测值predict和真实标签y
        if predict == y:
            right += 1
        else:
            # 将识别错误的样例打印了出来
            img_path = test_dataset.samples[i][0]
            print(f"wrong case: predict = {predict} y = {y} img_path = {img_path}")

    # 计算出测试效果
    sample_num = len(test_dataset)
    acc = right * 1.0 / sample_num
    print("test accuracy = %d / %d = %.3lf" % (right, sample_num, acc))
###########################################################

```


## MNIST-CHATGPT -- 

```
# CHATGPT写的一个手写数字识别
python3 train_mnist.py
# 得到模型mnist_cnn.pth

python3 test_mnist.py
准确率：99.20%

# 豆包写的一个手写数字识别，但是没写main函数，无法做模型的测试
pythno3 DOUBAO_mnist.py
得到模型DOUBAO_mnist.pth

# 改写豆包写的一个手写数字识别，改成main函数，能做模型的测试
python3 DOUBAO_mnist_0.py
# 得到模型DOUBAO_mnist_0.pth

python3 DOUBAO_mnist_test_0.py


```

  
  
## /pytorch_mnist.py  --  用PyTorch实现MNIST手写数字识别(非常详细)  
原文链接：https://nextjournal.com/gkoehler/pytorch-mnist    
​​​​​Keras版本：https://blog.csdn.net/sxf1061700625/article/details/117397030

PyTorch中构建一个简单的卷积神经网络，并使用MNIST数据集训练它识别手写数字。在MNIST数据集上训练分类器可以看作是图像识别的“hello world”。
MNIST包含70,000张手写数字图像: 60,000张用于培训，10,000张用于测试。图像是灰度的，28x28像素的，并且居中的，以减少预处理和加快运行。
使用PyTorch训练一个卷积神经网络来识别MNIST的手写数字。PyTorch是一个非常流行的深度学习框架，比如Tensorflow、CNTK和caffe2。但是与其他框架不同的是，PyTorch具有动态执行图，这意味着计算图是动态创建的。
先去官网上根据指南在PC上装好PyTorch环境，然后引入库。

```
import torch
import torchvision
from torch.utils.data import DataLoader
```

准备数据集
导入就绪后，我们可以继续准备将要使用的数据。但在那之前，我们将定义超参数，我们将使用的实验。在这里，epoch的数量定义了我们将循环整个训练数据集的次数，而learning_rate和momentum是我们稍后将使用的优化器的超参数。
```
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)
```

对于可重复的实验，必须为任何使用随机数产生的东西设置随机种子——如numpy和random!

需要数据集的dataloader。这就是TorchVision发挥作用的地方。它让我们用一种方便的方式来加载MNIST数据集。我们将使用batch_size=64进行训练，并使用size=1000对这个数据集进行测试。下面的Normalize()转换使用的值0.1307和0.3081是MNIST数据集的全局平均值和标准偏差，这里我们将它们作为给定值。

TorchVision提供了许多方便的转换，比如裁剪或标准化。

```
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)
```
运行上面的程序后，会自动将数据集下载到目录下的data文件夹。





```
pip3 install matplotlib
python3 pytorch_mnist.py


```
两个结论:
1. 从检查点内部状态继续按预期工作。
2. 我们似乎仍然没有遇到过拟合问题!看起来我们的dropout层做了一个很好的规范模型。
使用PyTorch和TorchVision构建了一个新环境，并使用它从MNIST数据集中对手写数字进行分类，希望使用PyTorch开发出一个良好的直觉。对于进一步的信息，官方的PyTorch文档确实写得很好，论坛也很活跃!

## /pytorch_mnist.py  --  用PyTorch实现MNIST手写数字识别（最新，非常详细）

```
# 详解

```




## /MNIST-CHATGPT  --   MNIST-CHATGPT
```
用于 加载保存的模型并对单张图片进行预测。这张图片可以是 MNIST 格式的（28x28 灰度图），或者是从外部导入的一张手写数字图像。
python predict_image.py example_digit.png


```

## /MNIST-SMG  --  尚码哥

```
MNIST-SMG/CNN1.py
CNN1.py使用 TensorFlow 1.x 风格实现的一个简单 CNN 卷积神经网络，用于识别 MNIST 手写数字图片。Cnn实现minist代码分类–tf.nn实现。

完全兼容 TensorFlow 2.x；
使用 tf.keras 构建高阶模型，清晰易懂；
保留原始架构（Conv → Pool → Conv → Pool → Dense → Dropout → Output）；
自动绘图，展示准确率与损失随 epoch 的变化。

pip3 uninstall tensorflow -y
pip3 install tensorflow-cpu


```

参考资料：
tensorflow 自己制作Mnist数据集，用训练好的模型来测试准确度。手把手教你实现！教你怎么调用自己已经训练好的模型！清晰易懂！加深你对卷积CNN的理解    https://blog.csdn.net/weixin_41146894/article/details/110010179  
深度学习入门！四种方式实现minist分类！全部详细代码实现！Cnn(卷积神经网络，两种方式)，感知机（Bp神经网络），逻辑回归！代码详细注释！！minist数据集该怎么使用？    https://blog.csdn.net/weixin_41146894/article/details/109779806  



