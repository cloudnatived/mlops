# https://www.sitepoint.com/best-sorting-algorithms/
def comb_sort(items):
    if isinstance(items, np.ndarray):
        items = items.copy().tolist()  # 复制并转换为列表
    gap = len(items)
    shrink = 1.3
    swapped = True
    
    while gap > 1 or swapped:
        # 计算下一个间隙
        gap = max(1, int(gap / shrink))  # 使用浮点数除法并取整
        swapped = False
        
        # 比较相隔gap的元素
        for i in range(len(items) - gap):
            if items[i] > items[i + gap]:
                items[i], items[i + gap] = items[i + gap], items[i]
                swapped = True
                
        #print(f"间隙: {gap}, 数组状态: {items[:10]}...")  # 打印前10个元素
        print(f"间隙: {gap}, 数组状态: {items}...")  # 打印数组
    
    return items

# 测试
import numpy as np
np.random.seed(42)  # 设置随机种子以便结果可重现
array = np.random.randint(0, 10000, size=100)  # 使用较小的数组便于查看
print("排序前:", array)

sorted_items = comb_sort(array)
print("排序后:", sorted_items)
