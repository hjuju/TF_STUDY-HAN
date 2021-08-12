from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# 1. 로스와 R2로 평가
# MIN MAX와 스탠다드 결과들 명시


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

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=60) # train 309, test 133

scaler = QuantileTransformer()
# scaler = MaxAbsScaler()
# scaler = PowerTransformer()
# scaler = RobustScaler()
scaler.fit(x_train) # xtrain에 대해서만 스케일러 해줌
x_train = scaler.transform(x_train) # 전체 데이터로 스케일링 하면 과적합이 될 수 있기때문에 나누어서 스케일링 해줌
x_test = scaler.transform(x_test) # 비율에 맞춰서 스케일링



#2. 모델 구성
model = Sequential()
model.add(Dense(512, input_shape=(10,), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

optimizer = Adam(lr=0.001) 

es = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5) 
model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
start = time.time()
model.fit(x_train, y_train, epochs=1000, verbose=2, batch_size=16,validation_split=0.2,  shuffle=True, callbacks=[es, reduce_lr])
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
ic| loss: [3098.328857421875, 44.891075134277344]
ic| r2: 0.49399747798900706
ic| end: 4.975674629211426

모두 동일한 히든 레이어 

ic| r2: 0.5237891323534569
mse, R2

# StandartScarler
ic| loss: [5789.4814453125, 57.58549880981445]
ic| r2: 0.054492812279822944
ic| end: 8.076404809951782

model.fit(x_train, y_train, epochs=200, verbose=2, batch_size=32, validation_split=0.3, shuffle=True)

# MinMaxScaler
ic| loss: [3506.58544921875, 46.12666702270508]
ic| r2: 0.42732318795572477
ic| end: 4.481018543243408

model.fit(x_train, y_train, epochs=100, verbose=2, batch_size=32, validation_split=0.3, shuffle=True)

# QuantileTransformer
ic| loss: [2911.227294921875, 43.87248229980469]
ic| r2: 0.5245539161703192
ic| end: 1.3533804416656494

model.fit(x_train, y_train, epochs=10, verbose=2, batch_size=16, validation_split=0.3, shuffle=True)


# MaxAbsScaler

ic| loss: [3484.2333984375, 48.26594543457031]
ic| r2: 0.4309736285998824
ic| end: 1.1698718070983887

model.fit(x_train, y_train, epochs=10, verbose=2, batch_size=32, validation_split=0.3, shuffle=True)


# PowerTransformer

ic| loss: [3617.263671875, 47.10135269165039]
ic| r2: 0.4092478070732358
ic| end: 1.529919147491455

model.fit(x_train, y_train, epochs=20, verbose=2, batch_size=32, validation_split=0.3, shuffle=True)


# RobustScaler

ic| loss: [3907.24853515625, 49.15325927734375]
ic| r2: 0.3618890342096096
ic| end: 2.571096181869507

model.fit(x_train, y_train, epochs=20, verbose=2, batch_size=32, validation_split=0.3, shuffle=True)

'''