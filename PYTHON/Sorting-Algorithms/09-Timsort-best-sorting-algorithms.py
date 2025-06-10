# https://www.sitepoint.com/best-sorting-algorithms/
def insertion_sort(arr, left=0, right=None):
    if right is None:
        right = len(arr) - 1
        
    for i in range(left + 1, right + 1):
        key = arr[i]
        j = i - 1
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def merge(left, right):
    merged = []
    left_idx = right_idx = 0
    
    while left_idx < len(left) and right_idx < len(right):
        if left[left_idx] <= right[right_idx]:
            merged.append(left[left_idx])
            left_idx += 1
        else:
            merged.append(right[right_idx])
            right_idx += 1
            
    merged.extend(left[left_idx:])
    merged.extend(right[right_idx:])
    return merged

def timsort(arr):
    if len(arr) < 64:  # 对于小数组直接使用插入排序
        return insertion_sort(arr.copy())
    
    # 确定合适的min_run值（通常为32-64之间的2的幂）
    min_run = 32
    n = len(arr)
    
    # 对每个min_run长度的子数组进行插入排序
    for i in range(0, n, min_run):
        insertion_sort(arr, i, min(i + min_run - 1, n - 1))
    
    # 逐步合并已排序的子数组
    size = min_run
    while size < n:
        for start in range(0, n, size * 2):
            mid = start + size - 1
            end = min(start + size * 2 - 1, n - 1)
            
            # 提取左右子数组并合并
            left = arr[start:mid + 1]
            right = arr[mid + 1:end + 1]
            merged = merge(left, right)
            
            # 将合并结果放回原数组
            arr[start:start + len(merged)] = merged
            
        size *= 2
        
    return arr

# 测试
import numpy as np
np.random.seed(42)  # 设置随机种子以便结果可重现
items = np.random.randint(0, 100, size=20)  # 使用较小的数组便于查看
print("排序前:", items)

sorted_items = timsort(items.copy())  # 传递数组副本
print("排序后:", sorted_items)
print("原始数组:", items)  # 验证原始数组未被修改
