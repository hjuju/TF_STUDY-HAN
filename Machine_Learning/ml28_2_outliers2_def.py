import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

aaa = np.array([[1, 2, 1000, 3, 4, 6, 7, 8, 90, 100, 1000],
                [1000, 2000, 3, 4000, 5000, 6000, 7000, 8, 5000, 20000, 1001]])


aaa = aaa.transpose()

ic(aaa.shape)

def outlier(data_out):
    lis = []
    for i in range(data_out.shape[1]):
        quartile_1, q2, quartile_3 = np.percentile(data_out[:, i], [25, 50, 75])
        print("Q1 : ", quartile_1)
        print("Q2 : ", q2)
        print("Q3 : ", quartile_3)
        iqr = quartile_3 - quartile_1
        print("IQR : ", iqr)
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        print('lower_bound: ', lower_bound)
        print('upper_bound: ', upper_bound)

        m = np.where((data_out[:, i]>upper_bound) | (data_out[:, i]<lower_bound))
        n = np.count_nonzero((data_out[:, i]>upper_bound) | (data_out[:, i]<lower_bound))
        lis.append([i+1,'columns', m, 'outlier_num :', n])

    return np.array(lis)

outliers_loc = outlier(aaa)
print("outlier at :\n", outliers_loc)

 

# 2차원 리스트일때 이상치 출력 되게 하기

plt.boxplot(aaa)
plt.show()