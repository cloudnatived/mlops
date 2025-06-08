import torch
import torchvision
import torch.nn as nn
from model import LeNet        # 导入自定义的LeNet模型
import torch.optim as optim
import torchvision.transforms as transforms


def main():
    transform = transforms.Compose(                                                   # 数据预处理：转为张量并标准化
        [transforms.ToTensor(),                                                       # 将图像转为[0,1]的Tensor
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])                     # 标准化：减去均值除以标准差，适配CIFAR10数据分布

    # 加载训练数据集。CIFAR10数据集包含50000张训练图片，10个类别
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,               # 数据集存放目录，表示是数据集中的训练集
                                             download=True, transform=transform)      # 第一次运行时为True，下载数据集，下载完成后改为False，预处理过程
                                             #download=False, transform=transform) 

    # 加载训练集，实际过程需要分批次（batch）训练   
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,              # 导入的训练集，每批训练的样本数
                                               shuffle=True, num_workers=0)           # 是否打乱训练集，使用线程数，在windows下设置为0

    # 加载验证数据集。10000张验证图片（与训练集同分布，类别相同）
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,                 # 同训练集存储路径。标识为验证集   
                                           download=False, transform=transform)        # 已下载则设为False。应用相同预处理
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,                 # 一次性加载5000张验证图像
                                             shuffle=False, num_workers=0)             # 验证时不打乱顺序
    val_data_iter = iter(val_loader)                                                   # 提取一批验证数据用于实时评估。转换为迭代器。
    val_image, val_label = next(val_data_iter)                                         # 获取第一批验证数据
    
    # 初始化模型与优化器
    net = LeNet()                                                                      # 实例化LeNet模型
    loss_function = nn.CrossEntropyLoss()                                              # 交叉熵损失函数（适用于多分类）
    optimizer = optim.Adam(net.parameters(), lr=0.001)                                 # Adam优化器：自适应学习率，适合深度学习任务

    # 模型训练主循环
    for epoch in range(5):                                                             # 训练5个完整轮次（epoch）。loop over the dataset multiple times
        running_loss = 0.0                                                             # 累计批次损失
        for step, data in enumerate(train_loader, start=0):                            # 遍历训练数据加载器（按批次迭代）

            inputs, labels = data                                                      # 解包批次数据：inputs为图像，labels为真实标签
            optimizer.zero_grad()                                                      # 优化器梯度清零（避免梯度累积）
            outputs = net(inputs)                                                      # 前向传播：模型预测
            loss = loss_function(outputs, labels)                                      # 计算损失：预测值与真实标签的交叉熵
            loss.backward()                                                            # 反向传播：计算梯度
            optimizer.step()                                                           # 参数更新：优化器执行一步更新

            running_loss += loss.item()                                                # print statistics。累计损失值，用于周期性输出
            if step % 500 == 499:                                                      # print every 500 mini-batches。每500个批次输出一次训练状态
                with torch.no_grad():                                                  # 验证模式：关闭梯度计算，减少内存消耗
                    outputs = net(val_image)                                           # 输出形状：[batch, 10]。用验证数据进行预测
                    predict_y = torch.max(outputs, dim=1)[1]                           # 提取概率最大的类别索引作为预测结果
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)  # 计算准确率：正确预测数 / 总样本数

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %          # 打印训练进度：轮次、批次、平均损失、验证准确率
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0                                                 # 重置累计损失

    print('Finished Training')

    save_path = './Lenet.pth'                                                          # 模型保存路径
    torch.save(net.state_dict(), save_path)                                            # 仅保存模型参数（而非整个模型），文件更小且更灵活


if __name__ == '__main__':
    main()
