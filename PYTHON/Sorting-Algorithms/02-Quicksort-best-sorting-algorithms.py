# https://www.sitepoint.com/best-sorting-algorithms/
def quick_sort(items):
    if len(items) > 1:
        pivot = items[0]
        left = [i for i in items[1:] if i < pivot]
        right = [i for i in items[1:] if i >= pivot]
        return quick_sort(left) + [pivot] + quick_sort(right)
    else:
        return items

#items = [6,20,8,19,56,23,87,41,49,53]
import numpy as np
array = np.random.randint(0, 10000, size=100)    # 生成一个100个数字的数组，数字在0-10000之间
print(array)

items = (array)
print(quick_sort(items))
