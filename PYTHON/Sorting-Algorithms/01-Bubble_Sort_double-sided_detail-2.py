import numpy as np
import time
import matplotlib.pyplot as plt

def bubble_sort(items, show_process=False):
    """
    优化的冒泡排序算法
    参数:
        items: 待排序的可迭代对象
        show_process: 是否显示排序过程
    返回:
        排序后的列表
    """
    items = list(items)
    n = len(items)
    sort_steps = []  # 记录每轮排序后的状态
    
    for i in range(n - 1):
        swapped = False  # 交换标志
        for j in range(n - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
                swapped = True
        
        # 记录当前排序状态
        sort_steps.append(items.copy())
        
        # 提前终止：如果没有交换，说明已排序
        if not swapped:
            print(f"提前终止，共进行{i+1}轮排序")
            break
    
    # 显示排序过程
    if show_process:
        print("排序过程:")
        for i, step in enumerate(sort_steps):
            print(f"第{i+1}轮: {step}")
    
    return items

def analyze_performance():
    """分析冒泡排序在不同数据规模下的性能"""
    sizes = [100, 200, 300, 400, 500]
    times = []
    
    for size in sizes:
        # 生成随机数组
        array = np.random.randint(0, 10000, size=size)
        # 测量排序时间
        start_time = time.time()
        bubble_sort(array)
        end_time = time.time()
        times.append(end_time - start_time)
    
    # 绘制性能曲线
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'o-', color='blue')
    plt.title('冒泡排序性能分析')
    plt.xlabel('数据规模')
    plt.ylabel('排序时间(秒)')
    plt.grid(True)
    plt.show()

# 示例用法
if __name__ == "__main__":
    # 示例1: 排序并显示过程
    #array1 = np.random.randint(0, 100, size=10)
    array1 = np.random.randint(0, 10000, size=100)
    print("示例1:")
    print("原始数组:", array1)
    sorted_items1 = bubble_sort(array1, show_process=True)
    print("排序结果:", sorted_items1)
    
    # 示例2: 性能分析
    print("\n示例2: 性能分析")
    analyze_performance()
