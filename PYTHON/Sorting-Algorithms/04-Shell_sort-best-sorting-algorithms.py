# https://www.sitepoint.com/best-sorting-algorithms/
def shell_sort(items):
    sublistcount = len(items)//2
    while sublistcount > 0:
        for start in range(sublistcount):
            gap_insertion_sort(items, start, sublistcount)
        sublistcount = sublistcount // 2
    return items

def gap_insertion_sort(items, start, gap):
    for i in range(start+gap, len(items), gap):
        currentvalue = items[i]
        position = i
        while position >= gap and items[position-gap] > currentvalue:
            items[position] = items[position-gap]
            position = position-gap
        items[position] = currentvalue

#items = [6,20,8,19,56,23,87,41,49,53]
import numpy as np
array = np.random.randint(0, 10000, size=100)    # 生成一个100个数字的数组，数字在0-10000之间
print(array)

items = (array)
print(shell_sort(items))
