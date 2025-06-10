# https://www.sitepoint.com/best-sorting-algorithms/
def selection_sort(items):
    for i in range(len(items)):
        min_idx = i
        for j in range(i+1, len(items)):
            if items[min_idx] > items[j]:
                min_idx = j
        items[i], items[min_idx] = items[min_idx], items[i]
        print(f"第{i+1}轮排序结果: {items}") # 打印每轮排序后的结果（可选）
    return items

#items = [6,20,8,19,56,23,87,41,49,53]
import numpy as np
array = np.random.randint(0, 10000, size=100)    # 生成一个100个数字的数组，数字在0-10000之间
print(array)

items = (array)
print(selection_sort(items))
