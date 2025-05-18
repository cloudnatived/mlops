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
        transforms.Grayscale(num_output_channels=1),      # transform 功能1.转换为单通道灰度图
        transforms.ToTensor()                             # transform 功能2.转换为张量
    ])

    # 使用ImageFolder所数，读取数据文件夹，构建数据集dataset
    # 这个函数会将保存数据的文件夹的名字，作为数据的标签，组织数据
    # 例如，对于名字为"3"的文件夹
    # 会将"3"就会作为文件夹中图像数据的标签，和图像配对，用于后续的训练，使用起来非常的方便
    
    # 读入并构造数据集
    train_dataset = datasets.ImageFolder(root='./mnist_images/train', transform=transform)
    print("train_dataset length: ", len(train_dataset))

    # 使用trainloader,实现小批量的数据读取
    # 这里设置小批量的大小，batch size=64。也就是每个批次，包括64个数据
    # 小批量的数据读入
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print("train_loader length: ", len(train_loader))
    # 60000个训练数据，如果每个小批量，读入64个样本，那么60000个数据会被分成938组
    # 计算938*64=60032，这说明最后一组，会不够64个数据

    model = Network()                              # 模型本身，它就是我们设计的神经网络
    optimizer = optim.Adam(model.parameters())     # 优化模型中的参数
    criterion = nn.CrossEntropyLoss()              # 分类问题，使用交叉熵损失误差

    # 进入模型的迭代循环
    for epoch in range(10):  # 外层循环，代表了整个训练数据集的遍历次数，整个训练集要循环多少轮，设置成10次、20次或者100次都是可以的

        # 内存循环使用train_loader，进行小批量的数据读取，每一次循环，都会取出64个图像数据，作为一个小批量batch
        for batch_idx, (data, label) in enumerate(train_loader):
            # 内层每循环一次，就会进行一次梯度下降算法
            # 包括了5个步骤:
            output = model(data)                   # 1.计算神经网络的前向传播结果
            loss = criterion(output, label)        # 2.计算output和标签label之间的损失loss
            loss.backward()                        # 3.使用backward计算梯度
            optimizer.step()                       # 4.使用optimizer.step更新参数
            optimizer.zero_grad()                  # 5.将梯度清零
                                                   # 这5个步骤，是使用pytorch框架训练模型的定式，简单记忆就可以了

            if batch_idx % 100 == 0:               # 每迭代100个小批量，就打印一次模型的损失，观察训练的过程
                print(f"Epoch {epoch + 1}/10 "
                      f"| Batch {batch_idx}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'mnist.pth')    # 保存模型
