# https://www.sitepoint.com/best-sorting-algorithms/
def radix_sort(items):
    max_length = False
    tmp, placement = -1, 1

    while not max_length:
        max_length = True
        buckets = [list() for _ in range(10)]

        for i in items:
            tmp = i // placement
            buckets[tmp % 10].append(i)
            if max_length and tmp > 0:
                max_length = False

        a = 0
        for b in range(10):
            buck = buckets[b]
            for i in buck:
                items[a] = i
                a += 1

        placement *= 10
    return items

#items = [6,20,8,19,56,23,87,41,49,53]
import numpy as np
array = np.random.randint(0, 10000, size=100)    # 生成一个100个数字的数组，数字在0-10000之间
print(array)

items = (array)
print(radix_sort(items))
