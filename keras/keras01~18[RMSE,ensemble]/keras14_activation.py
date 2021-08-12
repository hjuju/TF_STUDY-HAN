from operator import mod
import numpy as np
from numpy.matrixlib.defmatrix import matrix
# import pandas as pd
from sklearn.datasets import load_diabetes
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time


'''
과제 결과

ic| loss: 1954.767578125
ic| r2: 0.6763044229890984
ic| end: 6.143572568893433

'''



#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

ic(x.shape, y.shape) # x.shape: (442, 10), y.shape: (442,)

# ic(datasets.feature_names)
# # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# ic(datasets.DESCR)

# ic(y[:30]) # 30개 데이터 출력

# ic(np.min(y), np.max(y)) # 최소, 최대값 

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=10)




#2. 모델 구성
model = Sequential()
# 활성화 함수 -> 모든 레이어에 존재, 통상적으로 relu가 성능 좋고, 마지막에는 activation X 
# model.add(Dense(300, input_shape=(10,),activation='linear')) 
# model.add(Dense(290,activation='linear'))
# model.add(Dense(270,activation='linear'))
# model.add(Dense(1))


model.add(Dense(256, input_shape=(10,),activation='selu')) 
model.add(Dense(230,activation='selu'))
model.add(Dense(190,activation='selu'))
model.add(Dense(140,activation='selu'))
model.add(Dense(130,activation='selu'))
model.add(Dense(1))




#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
start = time.time()
model.fit(x_train, y_train, epochs=99, batch_size=32, validation_split=0.01, shuffle=True)
end = time.time() - start

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict(x_test)
# ic(y_predict)

r2 = r2_score(y_test, y_predict)
ic(r2)

ic(end)
