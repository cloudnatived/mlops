# https://blog.csdn.net/Alex_thomas/article/details/121239611
# 选择排序 - 通过位置来找元素
import numpy as np
import time

# 设置随机种子以便结果可复现
np.random.seed(42)
sortList = np.random.randint(0, 10000, size=100)

# 选择排序 - 通过位置来找元素
def selection_sort(arr):
    arr = arr.copy()  # 避免修改原数组
    n = len(arr)
    
    for i in range(n - 1):
        min_index = i  # 记录当前最小值的索引
        
        # 找出未排序部分的最小值索引
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
                
        # 如果找到更小的值，交换当前元素与最小值
        if min_index != i:
            arr[i], arr[min_index] = arr[min_index], arr[i]
            
    return arr

# 测试选择排序
print("排序前:", sortList[:10], "...")  # 只显示前10个元素

# 记录开始时间
start_time = time.time()
sorted_list = selection_sort(sortList)
end_time = time.time()

print("排序后:", sorted_list[:10], "...")
print(f"排序耗时: {end_time - start_time:.6f}秒")
