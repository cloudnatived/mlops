# https://blog.csdn.net/Alex_thomas/article/details/121239611
# 归并排序
def merge_sort(ary):
        # 当传的列表元素为1时 开始归来（此条件作为递归出口）
        if len(ary) <= 1:
                return ary
   
        median = int(len(ary)/2)    # 二分分解 一直分解到每组只有一个元素为止
        """按顺序执行，先执行左边的元素的切割，再执行右边元素的切割"""
        left = merge_sort(ary[:median])#切片前闭后开
        right = merge_sort(ary[median:])
        """从最低层开始合并，从下往上的过程  ------归来"""
        return merge(left, right)    # 合并数组，从最底层开始
    
def merge(left, right):
        '''合并操作，
        将两个有序数组left[]和right[]合并成一个大的有序数组'''
        new = []
        i = j = k = 0
        # 循环比较两个数列的数值大小，把小的元素存放到new列表中
        while(i < len(left) and j < len(right)):
                if left[i] < right[j]:
                        new.append(left[i])
                        i += 1
                else:
                        new.append(right[j])
                        j += 1
        """拼接多余的元素"""
        new = new + left[i:] + right[j:]
        return new
print(merge_sort([1,2,1,3,2,4,5,3,4,9,7,6]))
