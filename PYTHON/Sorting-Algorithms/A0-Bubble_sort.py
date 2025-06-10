# https://blog.csdn.net/Alex_thomas/article/details/121239611
# 冒泡排序 - 通过元素比较找位置
import numpy as np
import time

# 设置随机种子以便结果可复现
np.random.seed(42)
sortList = np.random.randint(0, 10000, size=100)
print("排序前:", sortList)

# 记录开始时间
start_time = time.time()

# 冒泡排序核心逻辑
for i in range(len(sortList) - 1):
    for j in range(len(sortList) - 1 - i):
        if sortList[j] > sortList[j + 1]:
            # 交换元素位置
            sortList[j], sortList[j + 1] = sortList[j + 1], sortList[j]

# 计算排序耗时
end_time = time.time()
print(f"排序耗时: {end_time - start_time:.6f}秒")
print("排序后:", sortList)
