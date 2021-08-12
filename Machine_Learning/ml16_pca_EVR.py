import re
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston, load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from icecream import ic


#1. 데이터

datasets = load_diabetes()

x = datasets.data # x.shape: (506, 13)
y = datasets.target

pca = PCA(n_components=10) # 컬럼을 n개로 압축(삭제가 아님)
x = pca.fit_transform(x)
# ic(x, x.shape) # (442,7)

pca_EVR = pca.explained_variance_ratio_  # 0.95 정도로 잡는다면 손실이 크지 않음 -> 0.95 까지 n_components값을 조절
# 압축한 결과의 중요도(중요한 순서대로 몰아버림) -> pca에서 적용한 숫자만큼 총합이 달라짐(전체 컬럼일 때(압축안했을때) 1.0 -> 이때 차원축소는 아님)
ic(pca_EVR)
ic(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR) # EVR의 누적합계로 확인하여 컬럼 축소할 수치 변경
ic(cumsum)
'''
ic| cumsum: array([0.40242142, 0.55165324, 0.67224947, 0.76779711, 0.83401567,
                   0.89428759, 0.94794364, 0.99131196, 0.99914395, 1.        ])
'''

ic(np.argmax(cumsum >= 0.94)+1)
# np.argmax(cumsum >= 0.94)+1: n_components = 7 
x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, train_size=0.8, random_state=66)

ic(x.shape)

import matplotlib.pyplot as plt # cumsum 시각화
plt.plot(cumsum)
plt.grid()
plt.show()


#2. 모델
from xgboost import XGBRegressor
model = XGBRegressor()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측

results = model.score(x_test,y_test)
ic(results)

'''
diabets

ic| x.shape: (442, 10)
ic| results: 0.23802704693460175

PCA:8 -> EVR(0.991)
c| x.shape: (442, 8)
ic| results: 0.3366979304013662

'''