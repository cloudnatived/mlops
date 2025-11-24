# 这个程序的功能会先将MNIST数据下载下来，然后再保存为.png的格式。

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm  # 增加进度条显示

# 注意，这里得到的train_data和test_data已经直接可以用于训练了！
# 不一定要继续后面的保存图像。
#train_data = MNIST(root='./data', train=True, download=True, transform=ToTensor())
train_data = MNIST(root='/Data/DEMO/MODEL/MNIST', train=True, download=True, transform=ToTensor())
#test_data = MNIST(root='./data', train=False, download=True, transform=ToTensor())
test_data = MNIST(root='/Data/DEMO/MODEL/MNIST', train=False, download=True, transform=ToTensor())

# 继续将手写数字图像“保存”下来
# 输出两个文件夹，train和test，分别保存训练和测试数据。
# train和test文件夹中，分别有0、1、2、3、4、5、6、7、8、9；
# 这10个子文件夹保存每种数字的数据

from torchvision.transforms import ToPILImage                                        # 张量转 PIL 图像，ToPILImage() 将张量转换为 PIL 图像对象（取值范围自动恢复为 [0, 255]，格式为灰度图）。
train_data = [(ToPILImage()(img), label) for img, label in train_data]               # 转换后的数据存储为列表，每个元素是 (PIL.Image, label) 的元组。
test_data = [(ToPILImage()(img), label) for img, label in test_data]

import os
import secrets
def save_images_with_progress(dataset, folder_name):
    root_dir = os.path.join('./mnist_images', folder_name)                           # 按folder_name（train或test）创建根目录。
    os.makedirs(root_dir, exist_ok=True)                                             # 创建根目录（如 ./mnist_images/train）
    #for i in range(len(dataset)):
    for i in tqdm(range(len(dataset)), desc=f"Saving {folder_name} images"):         # 增加进度条显示
        img, label = dataset[i]
        label_dir = os.path.join(root_dir, str(label))                               # 创建标签子目录（如 ./mnist_images/train/0）
        os.makedirs(label_dir, exist_ok=True)
        random_filename = secrets.token_hex(8) + '.png'                              # 使用secrets.token_hex(8)生成随机文件名，避免同名文件覆盖。8字节16进制字符串，如 "a1b2c3d4e5f6.png"
        img.save(os.path.join(label_dir, random_filename))                           # 保存图像

save_images_with_progress(train_data, 'train')
save_images_with_progress(test_data, 'test')
