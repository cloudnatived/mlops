def merge_sort(items):
    # 转换为列表处理（如果输入是NumPy数组）
    if isinstance(items, np.ndarray):
        items = items.tolist()

    if len(items) <= 1:
        return items.copy()  # 返回副本，避免修改原数组
    
    mid = len(items) // 2
    left = items[:mid]
    right = items[mid:]
    
    # 递归排序左右两部分
    left = merge_sort(left)
    right = merge_sort(right)
    
    # 合并已排序的两部分并输出过程
    merged = merge(left, right, items.copy())
    print(f"合并 {left} 和 {right} -> {merged}")
    return merged

def merge(left, right, original):
    merged = []
    left_idx = right_idx = 0
    
    # 比较左右数组元素并合并
    while left_idx < len(left) and right_idx < len(right):
        if left[left_idx] <= right[right_idx]:
            merged.append(left[left_idx])
            left_idx += 1
        else:
            merged.append(right[right_idx])
            right_idx += 1
    
    # 添加剩余元素
    merged += left[left_idx:]
    merged += right[right_idx:]
    
    return merged

# 测试
import numpy as np
np.random.seed(42)  # 设置随机种子以便结果可重现
items = np.random.randint(0, 10000, size=100)  # 生成10个0-99的随机数
print("排序前的数组:", items)

sorted_items = merge_sort(items)
print("排序后的数组:", sorted_items)
