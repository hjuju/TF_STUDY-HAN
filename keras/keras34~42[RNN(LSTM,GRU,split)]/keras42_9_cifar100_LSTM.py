import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import time
from icecream import ic

#1. data preprocessing
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)
ic(x_train.shape)
ic(x_test.shape)



scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32 * 32, 3)
x_test = x_test.reshape(10000, 32 * 32,  3)

one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray() 




#2. 모델링
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(32 * 32 ,3 )))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()

#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='acc', patience=10, mode='auto', verbose=1)
start = time.time()
model.fit(x_train, y_train, epochs=10, verbose=1, validation_split=0.2, batch_size=1024, shuffle=True, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. evaluating, prediction
loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0])
print('accuracy = ', loss[1])
ic(f'{걸린시간}분')

'''

LSTM

loss =  4.388662815093994
accuracy =  0.03530000150203705
ic| f'{걸린시간}분': '12.6분'


loss =  3.0406737327575684
accuracy =  0.3928000032901764

batch_size=64, validation_split=0.25
loss =  5.080616474151611
accuracy =  0.33799999952316284
ic| f'{걸린시간}분': '3.5분'

모델수정 / patience=7,epochs=100, batch_size=64, validation_split=0.25
loss =  2.777371406555176
accuracy =  0.376800000667572

DNN

loss =  3.594170331954956
accuracy =  0.1868000030517578
ic| f'{걸린시간}분': '2.3분'


                            
'''


