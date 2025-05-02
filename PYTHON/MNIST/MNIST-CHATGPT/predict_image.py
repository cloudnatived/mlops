import torch
from torchvision import transforms
from PIL import Image
from train_mnist import SimpleCNN  # 确保 train_mnist.py 中定义的 SimpleCNN 类可导入
import sys

def predict(image_path):
    # 图像预处理：转为灰度图 → 缩放至 28x28 → 转 Tensor → 标准化
    transform = transforms.Compose([
        transforms.Grayscale(),  # 保证为单通道
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 的均值与方差
    ])

    # 加载图片并预处理
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 添加 batch 维度 [1, 1, 28, 28]

    # 加载模型
    model = SimpleCNN()
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu"))
    model.eval()

    # 推理
    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output, dim=1).item()

    print(f"预测结果：{predicted}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python predict_image.py path_to_image")
    else:
        predict(sys.argv[1])

