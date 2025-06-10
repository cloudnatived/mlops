# https://blog.csdn.net/Alex_thomas/article/details/121239611
# 快速排序
 
def quick_sort(alist, first, last):
    """快速排序"""
    if first >= last:
        return 
        """选取基准元素 == 挖出基准元素"""
    mid_value = alist[first]
    """定义两个 下标 (指针) 指向数列两端"""
    low = first
    high = last
    while low < high:
        while low < high and alist[high] >= mid_value:
            high -= 1
        """ 打破循环的条件是 左边的下标所指向的元素小于基准元素"""
        if low < high:
                alist[low] = alist[high]
                """ 赋完值后下标右进一位"""
                low+=1
        while low < high and alist[low] < mid_value:
            low += 1
            """打破循环的条件 是右边的下标所指向的元素大于基准元素"""
        if low<high:
                alist[high] = alist[low]
                """ 赋完值后下标左进一位"""
                high-=1
    # 从大循环退出时，low == high
    """ 把基准元素插入 此时插入的位置在中间"""
    alist[low] = mid_value
    """对low左边的列表执行快速排序"""
    quick_sort(alist, first, low-1)
    """对low右边的列表执行快速排序"""
    quick_sort(alist, low+1, last)
 
# if __name__ == '__main__':
li = [0, 3, 2, 1, 5, 4, 7, 6, 10, 3]
print(li)
quick_sort(li, 0, len(li)-1)
print(li)
