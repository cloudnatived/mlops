# https://www.sitepoint.com/best-sorting-algorithms/
def insertion_sort(items):
    for i in range(1, len(items)):
        j = i
        while j > 0 and items[j-1] > items[j]:
            items[j-1], items[j] = items[j], items[j-1]
            j -= 1
    return items

#items = [6,20,8,19,56,23,87,41,49,53]
import numpy as np
array = np.random.randint(0, 10000, size=100)    # 生成一个100个数字的数组，数字在0-10000之间
print(array)

items = (array)
print(insertion_sort(items))
