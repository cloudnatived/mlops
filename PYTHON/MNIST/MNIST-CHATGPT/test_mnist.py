import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train_mnist import SimpleCNN

def test():
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)   # 读取测试数据集
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False) 

    model = SimpleCNN()                                                                             # 从同目录的train_mnist.py程序里读取所定义神经网络模型
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu"))                          # 加载刚刚训练好的模型文件
    model.eval()

    correct = 0                                                                                     # 保存正确识别的数量
    total = 0                                                                                       # 读取的图片总数
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)                                                                 # 将其中的数据x输入到模型
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"准确率：{100 * correct / total:.2f}%")

if __name__ == "__main__":
    test()

