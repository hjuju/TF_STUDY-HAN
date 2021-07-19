# (31,)로 끝나면 31,1,1로 가능 다차원으로 변환한것과 곱한것과 2차원의 쉐이프와 같아야함

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler
# 1. 로스와 R2로 평가
# MIN MAX와 스탠다드 결과들 명시

from tensorflow.keras.callbacks import EarlyStopping
from operator import mod
import numpy as np
from numpy.matrixlib.defmatrix import matrix
# import pandas as pd
from sklearn.datasets import load_boston
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

from tensorflow.python.keras import activations
from tensorflow.python.keras.layers.core import Dropout


#1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

# ic(x.shape, y.shape) # x.shape: (506, 13), y.shape: (506,)



# ic(datasets.feature_names)
# # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# ic(datasets.DESCR)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=60) # train 309, test 133

# scaler = QuantileTransformer()
scaler = StandardScaler()
#scaler = PowerTransformer()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

ic(x_train.shape[0], y.shape)




# 2. 모델 구성
model = Sequential()
model.add(Conv2D(32, kernel_size=2, padding='same', activation='relu', input_shape=(13,1,1)))  
model.add(Conv2D(64, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))                                      
model.add(Conv2D(128, 2, padding='same', activation='relu')) 
model.add(Dropout(0.2))                        
model.add(Conv2D(64, 2, padding='same', activation='relu'))
model.add(Conv2D(32, 2, padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())                                       
model.add(Dense(1))


#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1)
start = time.time()
model.fit(x_train, y_train, epochs=1000, verbose=1, validation_split=0.2, batch_size=8, shuffle=True, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict(x_test)
# ic(y_predict)

r2 = r2_score(y_test, y_predict)
ic(r2)

ic(f'{걸린시간}분')



'''
CNN
ic| loss: [14.31032943725586, 2.9922759532928467]
ic| r2: 0.8017186882057898
ic| f'{걸린시간}분': '0.6분'


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

'''