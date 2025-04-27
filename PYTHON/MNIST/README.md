## 用PyTorch实现MNIST手写数字识别

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



## mnist_network
三层神经网络训练手写数字识别


```

xhh_download_data.py
xhh_model.py
xhh_test.py
xhh_train.py

#下载数据集
pip3 install torchvision model
python3 download_data.py

#模型训练
python3 xhh-train.py

#模型测试
python3 xhh-test.py


```
