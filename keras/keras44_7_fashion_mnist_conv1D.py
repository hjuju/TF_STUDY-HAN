from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score
import pandas as pd



#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() # x_train.shape: (60000, 28, 28), y_train.shape: (10000, 28, 28)

ic(x_train.shape, y_train.shape)

x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)


# 전처리 

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28 * 28 , 1)
x_test = x_test.reshape(10000, 28 * 28 , 1)


ic(np.unique(y_train))

one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()





model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(28 * 28, 1 )))
model.add(Conv1D(128, 2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()

#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='acc', patience=10, mode='auto', verbose=1)
start = time.time()
model.fit(x_train, y_train, epochs=10, verbose=1, validation_split=0.2, batch_size=1024, shuffle=True, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. 평가, 예측

y_predict = model.predict(x_test)
loss = model.evaluate(x_test, y_test)
ic(loss[0])
ic(loss[1])
ic(f'{걸린시간}분')

'''
LSTM

ic| loss[0]: 1.529552698135376
ic| loss[1]: 0.4072999954223633
ic| f'{걸린시간}분': '11.3분'

CNN
loss =  0.3115113377571106
accuracy =  0.9218999743461609

DNN
loss =  0.3862316310405731
accuracy =  0.8654999732971191
ic| f'{걸린시간}분': '2.0분'

LSTM + Conv1D

'''

