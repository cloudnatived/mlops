# https://blog.csdn.net/Alex_thomas/article/details/121239611
# # 希尔排序（缩小增量排序）
def shell_sort(ary):
    # 增量分配
        gap=round(len(ary)/2)
        # 当（增量）gap=1时 整个文件恰被分成一组，gap=0时算法终止
        while gap>=1:
                """从后半部分开始遍历 即每组进行插入排序"""
                for i in range(gap,len(ary)):
                        while i-gap >= 0 and ary[i-gap]>ary[i]:
                                ary[i],ary[i-gap]=ary[i-gap],ary[i]
                                i-=gap
                """再次缩小增量"""
                gap = round(gap/2)
        return ary
print(shell_sort([1,3,5,2,4,3,6,4,3,7]))
