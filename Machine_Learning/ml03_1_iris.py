import numpy as np
from sklearn.datasets import load_iris
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

### 머신러닝(evaluate -> score)

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

# 1. 데이터
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)  # (150, 4), (150,)->(150, 3)
ic(y)   # (0,0,0, ... ,1,1,1, ... ,2,2,2, ...)

# *** 머신러닝에서는 1차원으로 받아들여야 해서 원핫인코딩, 투카테고리칼 하지 않음
# y = to_categorical(y)
# ic(y[:5])
# [0,0,0,0,0]
# [[1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]]
ic(y.shape)   # (150, )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# 1-2. 데이터 전처리
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델(머신러닝에서는 정의만 해주면 됨)
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression # 로지스틱회귀는 분류모델(회귀모델 X)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# model = SVC()
# ic| acc: 0.9555555555555556

# model = KNeighborsClassifier()
# ic| acc: 0.8888888888888888

# model = LogisticRegression()
# ic| acc: 0.9777777777777777

# model = DecisionTreeClassifier()
# ic| acc: 0.9111111111111111

model = RandomForestClassifier()
# ic| acc: 0.8888888888888888

# model = LinearSVC()
# ic| acc: 0.9111111111111111


# model = Sequential()
# model.add(Dense(128, activation='relu', input_shape=(4,)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(3, activation='softmax'))  # softmax : 다중분류   # 0,1,2  3개라서 3개로 나와야 함(150, 3)


# 3. 훈련(컴파일 포함되어 있어서 컴파일 할 필요 없음)
model.fit(x_train, y_train)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   # binary_crossentropy : 2진 분류   # metrics(결과에 반영X):평가지표

# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)

# model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[es])


# 4. 평가(evaluate 대신 score 사용함!!), 예측
results = model.score(x_test, y_test)       # accuracy
print('model.score:', results)
# results = model.evaluate(x_test, y_test)
# print('loss :', results[0])
# print('accuracy :', results[1])

from sklearn.metrics import r2_score, accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
ic(acc)

ic(y_test[:5])
y_predict = model.predict(x_test[:5])
ic(y_predict)   # 소프트맥스 통과한 값



