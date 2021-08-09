from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# 1. 로스와 R2로 평가
# MIN MAX와 스탠다드 결과들 명시


from operator import mod
import numpy as np
from numpy.matrixlib.defmatrix import matrix
# import pandas as pd
from sklearn.datasets import load_boston
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time


#1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

ic(x.shape, y.shape) # x.shape: (442, 10), y.shape: (442,)

# ic(datasets.feature_names)
# # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# ic(datasets.DESCR)

# ic(y[:30]) # 30개 데이터 출력

# ic(np.min(y), np.max(y)) # 최소, 최대값 

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=60) # train 309, test 133

# scaler = QuantileTransformer()
# scaler = MaxAbsScaler()
scaler = PowerTransformer()
# scaler = RobustScaler()
scaler.fit(x_train) # xtrain에 대해서만 스케일러 해줌
x_train = scaler.transform(x_train) # 전체 데이터로 스케일링 하면 과적합이 될 수 있기때문에 나누어서 스케일링 해줌
x_test = scaler.transform(x_test) # 비율에 맞춰서 스케일링



#2. 모델 구성
model = Sequential()
model.add(Dense(512, input_shape=(13,), activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

optimizer = Adam(lr=0.001) 

es = EarlyStopping(monitor='loss', patience=20, mode='auto', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', patience=10, mode='auto', verbose=1, factor=0.5) 

model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
start = time.time()
model.fit(x_train, y_train, epochs=400, verbose=2, batch_size=32, shuffle=True, callbacks=[es, reduce_lr])
end = time.time() - start

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict(x_test)
# ic(y_predict)

r2 = r2_score(y_test, y_predict)
ic(r2)

ic(end)

'''
ReduceLR
ic| loss: [7.76579475402832, 2.032741069793701]
ic| r2: 0.8923985750230131
ic| end: 15.358942031860352


ic| r2: 0.5237891323534569
mse, R2

model.add(Dense(512, input_shape=(13,), activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

# QuantileTransformer
ic| loss: [8.056320190429688, 2.04154109954834]
ic| r2: 0.8883731088039933
ic| end: 18.638136625289917

model.fit(x_train, y_train, epochs=500, verbose=2, batch_size=32, shuffle=True)


# MaxAbsScaler

model.add(Dense(512, input_shape=(13,), activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

ic| loss: [8.180001258850098, 2.150731325149536]
ic| r2: 0.8866594058457429
ic| end: 15.125555992126465

model.fit(x_train, y_train, epochs=400, verbose=2, batch_size=32, shuffle=True)


# PowerTransformer

model.add(Dense(512, input_shape=(13,), activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

ic| loss: [6.781839370727539, 1.9480767250061035]
ic| r2: 0.9060320691551077
ic| end: 18.702990531921387

model.fit(x_train, y_train, epochs=500, verbose=2, batch_size=32, validation_split=0.3, shuffle=True)


# RobustScaler

model.add(Dense(512, input_shape=(13,), activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

ic| loss: [8.540543556213379, 2.1857354640960693]
ic| r2: 0.88166378905062
ic| end: 14.968975305557251

model.fit(x_train, y_train, epochs=400, verbose=2, batch_size=32, validation_split=0.3, shuffle=True)

ReduceLR

'''