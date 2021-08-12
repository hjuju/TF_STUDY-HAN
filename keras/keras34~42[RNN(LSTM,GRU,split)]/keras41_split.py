import numpy as np
from icecream import ic

a = np.array(range(1,11))
size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):  # 10 - 5 + 1 = 6행 // 행의 개수가 정해짐
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a, size)

ic(dataset)

x = dataset[:, :5]
y = dataset[:, 3]

ic(x)
ic(y)

# 시계열 데이터는 x와 y를 분리를 해줘야함

'''
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]]

ic| x: array([[1, 2, 3, 4],
              [2, 3, 4, 5],
              [3, 4, 5, 6],
              [4, 5, 6, 7],
              [5, 6, 7, 8],
              [6, 7, 8, 9]])
              
ic| y: array([ 5,  6,  7,  8,  9, 10])

'''