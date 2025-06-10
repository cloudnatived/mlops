# https://www.sitepoint.com/best-sorting-algorithms/

def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items)-1-i):
            if items[j] > items[j+1]:
                items[j], items[j+1] = items[j+1], items[j]
        #print(f"第{i+1}轮排序结果: {items}", '=' * f, f) # 打印每轮排序后的结果（可选）
        #print('=' * f, f) # 打印每轮排序后的结果（可选）
        print(f"第{i+1}轮排序结果:")
        for value in items:
                print('=' * value, value)

    return items

#items = [6,20,8,19,56,23,87,41,49,53]
import numpy as np
array = np.random.randint(0, 100, size=10)    # 生成一个100个数字的数组，数字在0-10000之间
print(array)
items = (array)
print(bubble_sort(items))
