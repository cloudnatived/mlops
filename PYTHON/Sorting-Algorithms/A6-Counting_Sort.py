# https://blog.csdn.net/Alex_thomas/article/details/121239611
# 计数排序：
preList = [4,2,5,3,7,3,7,2,6]
maxsize = max(preList)
minsize = min(preList)
 
len_account = maxsize-minsize+1
offset = minsize
# 计数列表
list_account = [0]*len_account
# 初始化排序后的列表
laterList = [0]*len(preList)
 
for num in preList:   
        list_account[num-offset]+=1  #列表的初值都为0 可以直接用下标进行加减运算
index = 0
for i in range(0,len_account):
        # 计算列表中 数值为几 就循环几遍
        for j in range(0,list_account[i]):
                laterList[index]=i+offset
                index +=1
print("")
print(laterList)
