from enum import auto
from os import scandir
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler, OneHotEncoder
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
# import pandas as pd
from sklearn.datasets import load_diabetes
from icecream import ic
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

# save파일명: keras46_1_save_model.h5


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

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=60)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성

model = Sequential()
model.add(Dense(256, input_shape=(10,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# model.save('./_save/keras46_1_save_model_1.h5')
model.save_weights('./_save/keras46_1_save_weights_1.h5')

#model = load_model('./_save/keras46_1_save_model_1.h5') # Total params: 48,193 모델만 저장 됐기때문에 컴파일과 핏을 해야함(모델만 저장하고싶은 경우 모델 밑에다 세이브 모델)
#model = load_model('./_save/keras46_1_save_model_2.h5') #모델, 핏, 컴파일 저장했기 때문에 결과가 바로 나옴(가중치까지 저장하고싶은경우 컴파일과 핏 다음) Total params: 48,193

#model.summary()


#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
start = time.time()
model.fit(x_train, y_train, epochs=500, verbose=2, batch_size=8, validation_split=0.2, shuffle=True, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

# model.save('./_save/keras46_1_save_model_2.h5')
model.save_weights('./_save/keras46_1_save_weights_2.h5')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict(x_test)
# ic(y_predict)

r2 = r2_score(y_test, y_predict)
ic(r2)

ic(걸린시간)


'''
ic| r2: 0.5237891323534569
mse, R2

ic| loss: [2941.350341796875, 43.26279830932617]
ic| r2: 0.5196343793873097
ic| 걸린시간: 0.1

load model
ic| loss: [2942.417724609375, 43.317867279052734]
ic| r2: 0.5194600127912902
ic| 걸린시간: 0.1
'''