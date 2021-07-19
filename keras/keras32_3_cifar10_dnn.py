from operator import imod
from re import X
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import time
from icecream import ic

#1. data preprocessing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)
ic(x_train.shape)
ic(x_test.shape)



scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray() 




#2. 모델링
model = Sequential()
model.add(Dense(2048, activation='relu', input_shape=(32 * 32 * 3 ,))) 
model.add(Dense(1024, activation='relu'))                                      
model.add(Dropout(0.3)) 
model.add(Dense(512, activation='relu'))                   
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))                                          
model.add(Dense(128, activation='relu'))                   
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))                                     
model.add(Dense(10, activation='softmax')) 

#3. compiling, training
es = EarlyStopping(monitor='acc', patience=5, mode='auto', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                        metrics=['acc'])
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=128, 
                                validation_split=0.001, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. evaluating, prediction
loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0])
print('accuracy = ', loss[1])
ic(f'{걸린시간}분')

'''
CNN

loss =  0.8164052963256836
accuracy =  0.7332000136375427

DNN
loss =  1.6678853034973145
accuracy =  0.5084999799728394
ic| f'{걸린시간}분': '7.1분'


'''