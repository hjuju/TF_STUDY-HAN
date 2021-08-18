import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

aaa = np.array([[1,2,10000,3,4 ,6,7,8 ,90,100, 5000],
                [1000,2000,3,4000,5000,6000,7000,8,9000,10000,1001]])

aaa = aaa.transpose()

ic(aaa.shape)

from sklearn.covariance import EllipticEnvelope

outliers = EllipticEnvelope(contamination=.2)
'''
contaminationfloat, default=0.1
The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Range is (0, 0.5).
'''
outliers.fit(aaa)

results = outliers.predict(aaa) # 이상치를 -1로 표현해줌
ic(results)