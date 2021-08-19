# 라벨 0 112개 삭제# cancer로 만들기
# 지표는 f1

from imblearn.over_sampling import SMOTE
from numpy.lib.function_base import average
from pandas.core.series import Series
from sklearn import datasets
from sklearn.datasets import load_wine, load_breast_cancer
import pandas as pd
import numpy as np
from xgboost import XGBRFClassifier
from sklearn.model_selection import train_test_split
import time
import warnings
from xgboost.sklearn import XGBClassifier
warnings.filterwarnings('ignore')
from icecream import ic
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

ic(x.shape, y.shape)
# x.shape: (457, 29), y.shape: (457,)

y = np.array(y).reshape(569,1)
# print(y.shape)

# xy = np.concatenate((x,y), axis=1)
# print(dasd)

# xy = xy[xy[:, 30].argsort()]
# print(dasd)

x = x[112:,0:-1]
y = y[112:,-1]

ic(x.shape, y.shape) # x.shape: (569, 30), y.shape: (569,)

#############################################
##### 라벨 통합
#############################################

print("="*50)

ic(pd.Series(y).value_counts())

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66, stratify=y) # stratify=y
# stratify: class 비율(ratio)을 train / validation에 유지(한 쪽에 쏠려서 분배되는 것을 방지합니다) 
# 이 옵션을 지정해 주지 않고 classification 문제를 다룬다면, 성능의 차이가 많이 날 수 있음

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
ic(score) # score: 0.6587755102040816

y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
ic(f1)

'''
ic| score: 0.9565217391304348
ic| f1: 0.9486607142857143


ic| score2: 0.9782608695652174
ic| f1: 0.9748221127531472
'''

######################################### smote 적용 #########################################
print("="*40,'smote 적용',"="*40)

smote = SMOTE(random_state=66) # k_neighbors가 적어지면 성능이 떨어짐

start = time.time()
x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train) # 적은 데이터들을 가장 큰 데이터에 맞춰서 증폭
end = time.time() - start

print("걸린시간:", end)
ic(pd.Series(y_smote_train).value_counts())

ic(x_smote_train.shape, y_smote_train.shape) # (159, 13), y_smote_train.shape: (159,)

print("smote 전: ", x_train.shape, y_train.shape)
print("smote 후: ", x_smote_train.shape, y_smote_train.shape)
print("smote 전 레이블 값 분포: \n", pd.Series(y_train).value_counts())
print("smote 후 레이블 값 분포: \n", pd.Series(y_smote_train).value_counts())

model2 = XGBClassifier(n_jobs=-1)

model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')



score2 = model2.score(x_test, y_test)
ic(score2) 

y_pred = model2.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
ic(f1)

'''
ic| score2: 0.9736842105263158
ic| f1: 0.9712773998488284

stratify
ic| score2: 0.956140350877193
ic| f1: 0.9521289997480473
'''
