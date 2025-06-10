# https://www.sitepoint.com/best-sorting-algorithms/
def bucket_sort(items):
    buckets = [[] for _ in range(len(items))]
    for item in items:
        bucket = int(item/len(items))
        buckets[bucket].append(item)
    for bucket in buckets:
        bucket.sort()
    return [item for bucket in buckets for item in bucket]

#items = [6,20,8,19,56,23,87,41,49,53]
import numpy as np
array = np.random.randint(0, 10000, size=100)    # 生成一个100个数字的数组，数字在0-10000之间
print(array)

items = (array)


print(bucket_sort(items))
