# 高级优化：双向冒泡排序 (鸡尾酒排序)
def cocktail_sort(arr):
    n = len(arr)
    swapped = True
    start = 0
    end = n - 1
    
    while swapped:
        # 重置交换标志
        swapped = False
        
        # 从左到右排序，将大元素移到末尾
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        # 如果没有交换，提前终止
        if not swapped:
            break
            
        # 从右到左排序，将小元素移到开头
        end -= 1
        for i in range(end - 1, start - 1, -1):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
                
        start += 1
    return arr

# 测试
import numpy as np
array = np.random.randint(0, 10000, size=100)    # 生成一个100个数字的数组，数字在0-10000之间
print(array)

sortList = (array)
sorted_list = cocktail_sort(sortList)
print(sorted_list)
