# https://www.programiz.com/dsa/bubble-sort
# Bubble sort in Python

def bubbleSort(array):
    
  # loop to access each array element
  for i in range(len(array)):

    # loop to compare array elements
    for j in range(0, len(array) - i - 1):

      # compare two adjacent elements
      # change > to < to sort in descending order
      if array[j] > array[j + 1]:

        # swapping elements if elements
        # are not in the intended order
        temp = array[j]
        array[j] = array[j+1]
        array[j+1] = temp


#data = [-2, 45, 0, 11, -9]
import numpy as np
array = np.random.randint(-10000, 10000, size=100)    # 生成一个100个数字的数组，数字在0-10000之间
print(array)

#bubbleSort(data)
bubbleSort(array)

print('Sorted Array in Ascending Order:')
#print(data)
print(array)
