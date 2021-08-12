import imp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from icecream import ic
import time

#1. data preprocessing
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape((60000, 28 * 28 * 1))
x_test = x_test.reshape((10000, 28 * 28 * 1))

ic(np.unique(y_train))

scaler = MinMaxScaler()
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 
# print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
ic(y_train.shape)           
ic(y_test.shape) 


#2. modeling
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(28 * 28,))) 
model.add(Dropout(0.3)) 
model.add(Dense(128, activation='relu'))                                      
model.add(Dense(128, activation='relu'))                   
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))                                          
model.add(Dense(64, activation='relu'))                   
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))                                     
model.add(Dense(10, activation='softmax')) 


#3. compiling, training

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_split=0.001, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. evaluating, prediction
loss = model.evaluate(x_test, y_test)


import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

#1 
plt.subplot(2,1,1) # 그림을 2개그리는데 1행1렬
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')


#2
plt.subplot(2,1,2) # 그림을 2개그리는데 1행1렬
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()

print('loss = ', loss[0])
print('accuracy = ', loss[1])
ic(f'{걸린시간}분')

'''
CNN
loss =  0.3115113377571106
accuracy =  0.9218999743461609

DNN
loss =  0.3862316310405731
accuracy =  0.8654999732971191
ic| f'{걸린시간}분': '2.0분'

'''