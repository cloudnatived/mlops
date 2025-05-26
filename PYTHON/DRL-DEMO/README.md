

# MNIST 数据集
  

## /MNIST-XHH   -- 小黑黑 mnist_network
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




## /MNIST-CHATGPT -- 

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





## /LeNet-Pytorch   -- 
参考资料：Pytorch官方demo（LeNet）  https://zhuanlan.zhihu.com/p/1887640705884746376
pytorch图像分类篇：2.pytorch官方demo实现一个分类器(LeNet)    https://blog.csdn.net/m0_37867091/article/details/107136477


  
## /AlexNet-flower-Pytorch -- 使用pytorch搭建AlexNet并训练花分类数据集
参考资料：pytorch图像分类篇：3.搭建AlexNet并训练花分类数据集    https://blog.csdn.net/m0_37867091/article/details/107150142

数据集下载地址，数据集包含 5 中类型的花，每种类型有600~900张图像不等。
http://download.tensorflow.org/example_images/flower_photos.tgz



