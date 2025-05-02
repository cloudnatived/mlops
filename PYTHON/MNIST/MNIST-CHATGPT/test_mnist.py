import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train_mnist import SimpleCNN

def test():
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SimpleCNN()
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu"))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"准确率：{100 * correct / total:.2f}%")

if __name__ == "__main__":
    test()

