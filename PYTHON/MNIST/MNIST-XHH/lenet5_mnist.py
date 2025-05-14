# From AI doubao
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 定义LeNet-5架构的神经网络类
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 第一卷积层：输入1通道（灰度图像），输出6通道，卷积核大小为5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 第一池化层：最大池化，池化窗口大小为2x2，步幅为2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二卷积层：输入6通道，输出16通道，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 第二池化层：最大池化，池化窗口大小为2x2，步幅为2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第一个全连接层：输入维度是16*4*4，输出维度是120
        #self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 第二个全连接层：输入维度是120，输出维度是84
        self.fc2 = nn.Linear(120, 84)
        # 第三个全连接层：输入维度是84，输出维度是10，对应10个类别
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 前向传播函数定义网络的数据流向
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 4 * 4)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义数据变换和加载MNIST数据集
transform = transforms.Compose([
    transforms.Resize((32, 32)),                 # 将图像调整为32x32，符合LeNet-5输入要求
    transforms.ToTensor(),                       # 转换为张量
    transforms.Normalize((0.1307,), (0.3081,))   # MNIST数据集的均值和标准差
])


# 训练数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

# 测试数据集
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# 可视化训练数据
def visualize_samples(loader, title):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    fig.suptitle(title, fontsize=14)
    
    for i, ax in enumerate(axes.flat):
        img = images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

# 可视化训练样本
visualize_samples(train_loader, 'Training Samples')

# 检查是否有GPU可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化LeNet-5模型以及定义损失函数和优化器
net = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，用于分类问题
optimizer = optim.Adam(net.parameters(), lr=0.001)  # Adam优化器，学习率为0.001
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# 训练和验证历史记录
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

# 训练循环
def train(net, train_loader, criterion, optimizer, device, epoch):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()                       # 清零梯度
        outputs = net(inputs)                       # 前向传播
        loss = criterion(outputs, labels)           # 计算损失
        loss.backward()                             # 反向传播，计算梯度
        optimizer.step()                            # 更新权重

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if i % 100 == 99:  # 每100个批次打印一次
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc)
    
    return epoch_loss, epoch_acc

# 验证函数
def validate(net, val_loader, criterion, device):
    net.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100.0 * correct / total
    val_loss_history.append(epoch_loss)
    val_acc_history.append(epoch_acc)
    
    return epoch_loss, epoch_acc

# 训练主循环
best_val_acc = 0.0
num_epochs = 10  # 增加训练轮数

for epoch in range(num_epochs):
    train_loss, train_acc = train(net, train_loader, criterion, optimizer, device, epoch)
    val_loss, val_acc = validate(net, test_loader, criterion, device)
    
    # 学习率调整
    scheduler.step(val_loss)
    
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    print('-' * 50)
    
    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(net.state_dict(), 'best_lenet5.pth')
        print(f'Model saved with accuracy: {best_val_acc:.2f}%')

print("Finished Training")

# 加载最佳模型进行最终测试
net.load_state_dict(torch.load('best_lenet5.pth'))
net.eval()

# 测试模型
correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in test_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)  # 前向传播
        _, predicted = torch.max(outputs, 1)  # 找到最大概率的类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# 整体准确率
accuracy = 100 * correct / total
print(f"Final Accuracy on the test set: {accuracy}%")

# 每类准确率
for i in range(10):
    print(f'Accuracy of {i}: {100 * class_correct[i] / class_total[i]:.2f}%')

# 可视化训练历史
def plot_training_history():
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

# 可视化训练历史
plot_training_history()

# 可视化错误预测
def visualize_misclassifications(net, test_loader, device, num_samples=10):
    net.eval()
    misclassified = []
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    misclassified.append((inputs[i].cpu(), labels[i].cpu(), predicted[i].cpu()))
            
            if len(misclassified) >= num_samples:
                break
    
    # 可视化错误分类
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('Misclassified Examples', fontsize=14)
    
    for i, (img, true_label, pred_label) in enumerate(misclassified[:10]):
        ax = axes.flat[i]
        img = img.squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True: {true_label.item()}, Pred: {pred_label.item()}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('misclassified_examples.png')
    plt.close()

# 可视化错误预测
visualize_misclassifications(net, test_loader, device)    
