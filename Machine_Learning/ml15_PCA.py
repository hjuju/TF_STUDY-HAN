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

pca = PCA(n_components=7) # 컬럼을 7개로 압축(삭제가 아님)
x = pca.fit_transform(x)
# ic(x, x.shape) # (442,7)

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, train_size=0.8, random_state=66)

ic(x.shape)



#2. 모델
from xgboost import XGBRegressor
model = XGBRegressor()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측

results = model.score(x_test,y_test)
ic(results)

'''
boston

# ic| results: 0.9221188601856797

# PCA: ic| results: 0.8874143051779056

caner
ic| x.shape: (569, 30)
ic| results: 0.8186041948790475


ic| x.shape: (569, 18)
PCA : ic| results: 0.7517869652953406

diabets

ic| x.shape: (442, 10)
ic| results: 0.23802704693460175

PCA: 9
ic| x.shape: (442, 9)
ic| results: 0.33872988295954165

PCA: 7
ic| x.shape: (442, 7)
ic| results: 0.3210924574289413

'''