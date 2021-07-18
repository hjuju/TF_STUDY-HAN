import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from icecream import ic
import time

#1. 데이터 전처리
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape((50000, 32, 32, 3))/255
x_test = x_test.reshape((10000, 32, 32, 3))/255

# print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (50000, 100)
y_test = one.transform(y_test).toarray() # (10000, 100)



#2. 모델링
model = Sequential()
model.add(Conv2D(256, kernel_size=(2, 2),                          
                        padding='same', activation='relu', 
                        input_shape=(32, 32, 3))) 
model.add(Conv2D(256, (2, 2), padding='same', activation='relu'))                   
model.add(MaxPooling2D())                                         
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))                   
model.add(MaxPooling2D())                                         
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))    
model.add(MaxPooling2D())                                         
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))                   
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Flatten())                                              
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                        metrics=['acc'])
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=256, 
                                validation_split=0.001, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. evaluating, prediction
loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0])
print('accuracy = ', loss[1])
ic(f'{걸린시간}분')

'''
loss =  7.162003993988037
accuracy =  0.32089999318122864
'''