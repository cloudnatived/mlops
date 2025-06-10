# https://blog.csdn.net/Alex_thomas/article/details/121239611
#插入排序
def insert_sort(relist):
        # range() 前闭后开 从第二个元素开始
        for i in range(1,len(relist)):
                '''从后往前开始依次遍历 若未排序的元素比排完序的元素小,
                    则排完序的元素往后移动,未排序的往前移'''
                for j in range(i,0,-1):
                        if relist[j] <= relist[j-1]:
                                relist[j],relist[j-1] = relist[j-1],relist[j]
        print(relist)
relist = [8,3,2,6,1,4,9,7]
insert_sort(relist)
