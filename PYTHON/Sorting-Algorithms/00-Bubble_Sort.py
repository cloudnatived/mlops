# 基础优化：添加交换标志
def bubble_sort(arr):
    n = len(arr)
    # 外层循环：n-1轮排序
    for i in range(n - 1):
        # 交换标志，用于提前终止
        swapped = False
        # 内层循环：每轮比较n-1-i次
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                # 交换元素
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # 如果本轮没有交换，说明数组已有序，提前终止
        if not swapped:
            break
    return arr

# 测试
import numpy as np
array = np.random.randint(0, 10000, size=100)    # 生成一个100个数字的数组，数字在0-10000之间
print(array)

sortList = (array)
#sorted_list = bubble_sort(sortList)
sorted_list = bubble_sort(array.copy())  # 使用copy()方法避免修改原始数组
print(sorted_list)
