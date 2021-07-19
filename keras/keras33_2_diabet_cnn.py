from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler
# 1. 로스와 R2로 평가
# MIN MAX와 스탠다드 결과들 명시

from tensorflow.keras.callbacks import EarlyStopping
from operator import mod
import numpy as np
from numpy.matrixlib.defmatrix import matrix
# import pandas as pd
from sklearn.datasets import load_diabetes
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

from tensorflow.python.keras import activations
from tensorflow.python.keras.layers.core import Dropout


#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

# ic(x.shape, y.shape) # x.shape: (506, 13), y.shape: (506,)



# ic(datasets.feature_names)
# # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# ic(datasets.DESCR)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=60) # train 309, test 133

# scaler = QuantileTransformer()
scaler = StandardScaler()
#scaler = PowerTransformer()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

ic(x_train.shape[1], y.shape)




# 2. 모델 구성
model = Sequential()
model.add(Conv2D(32, kernel_size=2, padding='same', activation='relu', input_shape=(10,1,1)))  
model.add(Conv2D(64, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))                                      
model.add(Conv2D(128, 2, padding='same', activation='relu')) 
model.add(Dropout(0.2))                        
model.add(Conv2D(64, 2, padding='same', activation='relu'))
model.add(Conv2D(32, 2, padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())                                       
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae'])
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
c| loss: [4421.59033203125, 56.89407730102539]
ic| r2: 0.27803290337899045
ic| f'{걸린시간}분': '0.5분'

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


