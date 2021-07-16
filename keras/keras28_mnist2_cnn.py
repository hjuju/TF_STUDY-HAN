import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from tensorflow.keras.datasets import mnist
from icecream import ic
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping
import time


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ic(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) 흑백데이터 명시x
# ic(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)


# 전처리


x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255 # 3차원 -> 4차원  // 데이터의 내용과 순서가 바뀌면 안됨
x_test = x_test.reshape(10000, 28, 28, 1)/255

ic(x_train.shape)
ic(x_test.shape)


ic(np.unique(y_train)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
# ic(y_train.shape)            # (60000,10)
# ic(y_test.shape)




#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2,2), padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(20, (2,2), activation='relu'))               
model.add(Conv2D(30, (2,2), activation='relu', padding='same'))                
model.add(MaxPooling2D())                                     
model.add(Conv2D(20, (2,2)))
model.add(Conv2D(10, (2,2)))                               
model.add(Flatten())                                          
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='sigmoid')) # 이진분류로 출력

#3. 컴파일, 훈련 metrics=['acc']
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(monitor='acc', patience=5, mode='auto', verbose=1)
model.fit(x_train, y_train, epochs=1000, verbose=1, batch_size=128, callbacks=[es])
걸린시간 = (time.time() - start) /60

#4. 평가, 예측 predict 필요x acc로 판단

loss = model.evaluate(x_test, y_test)
ic('loss:', loss[0])
ic('accuracy', loss[1])
ic(걸린시간)
# ic(loss)

'''
ic| 'loss:', loss[0]: 0.00993986614048481
ic| 'accuracy', loss[1]: 0.9900000095367432
'''