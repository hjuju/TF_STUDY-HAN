from operator import imod
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import time
from icecream import ic


#1. data preprocessing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
ic(x_train.shape, x_test.shape)

x_train = x_train.reshape((50000, 32, 32, 3))/255
x_test = x_test.reshape((10000, 32, 32, 3))/255

one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray() 




#2. 모델링
model = Sequential()
model.add(Conv2D(256, kernel_size=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 3))) 
model.add(Conv2D(256, (2, 2), padding='same', activation='relu'))                   
model.add(MaxPooling2D())                                         
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))                   
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))    
model.add(MaxPooling2D())                                         
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))                   
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Flatten())                                              
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. compiling, training

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

optimizer = Adam(lr=0.001) 

es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.05) 
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_split=0.001, callbacks=[es, reduce_lr])
걸린시간 = round((time.time() - start) /60,1)

#4. evaluating, prediction
loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0])
print('accuracy = ', loss[1])
ic(f'{걸린시간}분')

'''
ReduceLR
loss =  1.5461745262145996
accuracy =  0.7699999809265137

loss =  0.8164052963256836
accuracy =  0.7332000136375427
'''