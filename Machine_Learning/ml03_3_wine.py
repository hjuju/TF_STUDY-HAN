# 모델 구성하고 완료

from sys import version_info
import sklearn
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from icecream import ic

# acc 0.8 이상 만들 것!!

datasets = load_wine()

x = datasets.data
y = datasets.target

print(np.shape(x))
print(np.shape(y))

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=70)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.svm import LinearSVC, SVC # 먹히는지 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

'''
model = LinearSVC()
acc: 0.9814814814814815


model =SVC()
acc: 0.9814814814814815


model = KNeighborsRegressor()
X


model = KNeighborsClassifier()
acc: 0.9814814814814815

model = LogisticRegression()
acc: 0.9814814814814815

model = KNeighborsClassifier()
acc: 0.9814814814814815


model = DecisionTreeClassifier()
acc: 0.9444444444444444

model = DecisionTreeRegressor()
acc: 0.9814814814814815

model = RandomForestRegressor()
X
'''
model = RandomForestClassifier()
acc: 0.9814814814814815


#3. 컴파일, 훈련


model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
ic(result)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
ic(acc)

# # ic| 'loss:', loss[0]: 0.25041162967681885
# # ic| 'accuracy', loss[1]: 0.9722222089767456

