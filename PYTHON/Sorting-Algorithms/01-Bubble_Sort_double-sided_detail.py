def cocktail_sort(arr):
    arr = arr.copy()  # 创建数组副本，避免修改原数组
    n = len(arr)
    swapped = True
    start = 0
    end = n - 1
    iteration = 0

    print(f"初始数组: {arr}")
    
    while swapped:
        # 重置交换标志
        swapped = False
        iteration += 1

        # 从左到右排序，将大元素移到末尾
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        print(f"遍历 {iteration} (左→右): {arr}")
        
        # 如果没有交换，说明数组已经有序
        if not swapped:
            break

        # 从右到左排序，将小元素移到开头
        end -= 1
        swapped = False  # 重置标志用于反向遍历
        for i in range(end, start - 1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        iteration += 1
        print(f"遍历 {iteration} (右→左): {arr}")

        start += 1
        
    return arr

# 测试
import numpy as np
np.random.seed(42)  # 设置随机种子，确保结果可重现
array = np.random.randint(0, 100, size=10)  # 生成10个0-99的随机数
print("排序前的数组:", array)

sorted_list = cocktail_sort(array)
print("最终排序结果:", sorted_list)
