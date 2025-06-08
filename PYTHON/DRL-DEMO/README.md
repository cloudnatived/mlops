
# DRL深度学习的入门任务

参考资料：  
MNIST手写体识别  https://docs.swanlab.cn/examples/mnist.html

LeNet 是一系列网络的合称，包括 LeNet-1、LeNet-4、LeNet-5 等，LeNet-5 是 LeNet 系列中最受欢迎和经典的版本。通常所说的 LeNet 和 LeNet-5 有以下区别：  
  
网络结构：   
LeNet：输入层（32x32 的图像）=> 卷积层 1（6 个 5x5 的卷积核）=> 池化层 1（2x2 的最大池化）=> 卷积层 2（16 个 5x5 的卷积核）=> 池化层 2（2x2 的最大池化）=> 全连接层 1（120 个神经元）=> 全连接层 2（84 个神经元）=> 输出层（10 个神经元）。   
LeNet-5：输入层（32x32 的图像）=> 卷积层 1（6 个 5x5 的卷积核）=> 池化层 1（2x2 的最大池化）=> 卷积层 2（16 个 5x5 的卷积核）=> 池化层 2（2x2 的最大池化）=> 卷积层 3（120 个 5x5 的卷积核）=> 全连接层 1（84 个神经元）=> 输出层（10 个神经元）。可以看出，LeNet-5 比 LeNet 多了一个卷积层，少了一个全连接层。  
  
复杂度与性能：一般来说，LeNet-5 由于增加了卷积层，能够提取更复杂的特征，在计算机视觉任务上理论上能有更好的表现。因为更深的网络结构可以学习到更高级的特征表示，对于图像中的细节和模式能够有更深入的理解和把握，从而提高分类等任务的准确性。而 LeNet 相对结构较简单，在面对复杂图像任务时可能表现会稍逊一筹。  
  
在实际应用中，如果是简单的图像识别任务，LeNet 可能就能够满足需求，并且由于其结构简单，训练速度可能会更快。但如果是较为复杂的图像任务，如高精度的手写数字识别或者其他类似的图像分类场景，LeNet-5 可能会是更好的选择，不过训练时间可能会相对较长。  

# MNIST 数据集

  
github上的一些参考：  
https://github.com/tm9161/MNIST/tree/main                      # 有一些基于MNIST手写数字的图像分类的示例。  
https://github.com/TiezhuXing01/LeNet5_in_PyTorch/tree/main    # 这是一个用 LeNet5 实现手写数字识别的项目。数据集用的是MNIST。  
https://github.com/lvyufeng/denoising-diffusion-mindspore      # 手撕 CNN 经典网络系列  



## 01-MNIST-XHH   -- 小黑黑 mnist_network
### 三层神经网络训练手写数字识别

```
# ubuntu 22.04.5 下的环境准备
#############################################################################################################
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
#############################################################################################################


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

# 下载数据集，并保存成图片，其它的手写数字识别程序没有这一步
python3 download_data.py

# 训练模型，讲模型保存成mnist.pth
python3 train.py

# 用测试数据集测试训练好的模型
python3 test.py

准确率：
test accuracy = 9801 / 10000 = 0.980

LeNet5_0.py

LeNet5_1.py

lenet5_mnist.py

```




## /03-MNIST-CHATGPT -- 

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
参考资料：用PyTorch实现MNIST手写数字识别(非常详细)   https://zhuanlan.zhihu.com/p/137571225  
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
参考资料：https://blog.csdn.net/qq_45588019/article/details/120935828    https://blog.csdn.net/qq_45588019/article/details/120935828  
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





## /04-MNIST-LeNet5   -- CIFAR10 图像

参考资料：https://blog.csdn.net/qq_37541097/category_9488472_2.html  
这个up主发了好多能动手实验的教程。太阳花的小绿豆。

参考资料： 
pytorch图像分类篇：2.pytorch官方demo实现一个分类器(LeNet)    https://blog.csdn.net/m0_37867091/article/details/107136477  

```
/LeNet-Pytorch/model.py
网络结构解析
1. 输入与输出
输入：3 通道彩色图像，尺寸为 32×32 像素（如 MNIST 数据集需扩展通道数）。
输出：10 维向量（对应 10 个类别），适用于分类任务。
2. 卷积层与池化层
第一层卷积：16 个 5×5 卷积核，提取基础特征（如边缘、纹理），输出尺寸 28×28（有效卷积：32-5+1=28）。
第一层池化：2×2 最大池化，降采样至 14×14，减少特征维度。
第二层卷积：32 个 5×5 卷积核，提取更复杂特征，输出尺寸 10×10（14-5+1=10）。
第二层池化：2×2 最大池化，降采样至 5×5，特征图数量保持 32。
3. 全连接层
fc1：将 32×5×5=800 维特征映射到 120 维，学习特征组合。
fc2：120→84，进一步压缩特征空间。
fc3：84→10，输出分类结果，对应 10 个类别（如 0-9 数字）。

关键操作说明
激活函数：使用 ReLU（Rectified Linear Unit）激活函数，解决梯度消失问题，加速收敛。
特征图尺寸计算：
卷积层：输出尺寸 = 输入尺寸 - 卷积核尺寸 + 1（假设步长 = 1，无填充）。
池化层：输出尺寸 = 输入尺寸 / 池化核大小（整除时）。
展平操作：x.view(-1, 32*5*5)将三维特征图（32,5,5）转换为一维向量（800 维），适配全连接层输入。

与 LeNet-5 的差异
当前代码实现的 LeNet 与经典 LeNet-5 的主要区别：
输入通道数：此处设为 3（彩色图像），而 LeNet-5 原始设计用于 1 通道灰度图像（如 MNIST 手写数字）。
网络深度：标准 LeNet-5 包含 3 个卷积层，此版本为 2 个卷积层，结构更简化。
池化方式：均采用最大池化，但 LeNet-5 可能使用平均池化或其他变体。

如需复现 LeNet-5，可添加第三层卷积层：
self.conv3 = nn.Conv2d(32, 64, 5)  # 新增卷积层

并在forward方法中添加对应操作。
```

```
/LeNet-Pytorch/test_model.py
python3 test_model.py 
模型在测试集上的整体准确率: 66.67%
plane 类别准确率: 73.50%
car 类别准确率: 72.20%
bird 类别准确率: 61.10%
cat 类别准确率: 46.60%
deer 类别准确率: 57.00%
dog 类别准确率: 56.50%
frog 类别准确率: 82.70%
horse 类别准确率: 61.50%
ship 类别准确率: 75.40%
truck 类别准确率: 80.20%


```

  
## /08-AlexNet-flower-Pytorch -- 使用pytorch搭建AlexNet并训练花分类数据集

```
参考资料：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test2_alexnet  
参考资料：pytorch图像分类篇：3.搭建AlexNet并训练花分类数据集    https://blog.csdn.net/m0_37867091/article/details/107150142  

数据集下载地址，数据集包含 5 中类型的花，每种类型有600~900张图像不等。  
wget http://download.tensorflow.org/example_images/flower_photos.tgz  


/08-AlexNet-flower-Pytorch/split_data.py # 用来拆分数据集

```


## /09-AlexNet-flower-tensorflow -- 使用tensorflow搭建AlexNet并训练花分类数据集

```
参考资料：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/tensorflow_classification/Test2_alexnet
参考资料：

数据集下载地址，数据集包含 5 中类型的花，每种类型有600~900张图像不等。  
wget http://download.tensorflow.org/example_images/flower_photos.tgz  

/09-AlexNet-flower-tensorflow/split_data.py # 用来拆分数据集


```



