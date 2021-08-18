import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

aaa = np.array([[1,2,10000,3,4 ,6,7,8 ,90,100, 5000],
                [1000,2000,3,4000,5000,6000,7000,8,9000,10000,1001]])

aaa = aaa.transpose()

ic(aaa.shape)

def outliers(data_out):
    for i in data_out:
        quantile_1, q2, quantile_3 = np.percentile(data_out, [25,50,75])
        print("1사분위: ", quantile_1)
        print("q2: ", q2)
        print("3사분위: ", quantile_3)
        iqr = quantile_3 - quantile_1
        lower_bound = quantile_1 - (iqr *1.5)
        upper_bound = quantile_3 + (iqr *1.5)
        return np.where((data_out>upper_bound) | (data_out<lower_bound))

outliers_loc = outliers(aaa)

print('이상치의 위치: ', outliers_loc)

# 2차원 리스트일때 이상치 출력 되게 하기

plt.boxplot(aaa)
plt.show()