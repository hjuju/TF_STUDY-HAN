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
ic(y_train.shape)           
ic(y_test.shape) 



# DNN 구해서 CNN비교
# DNN + GAP 구해서 CNN비교

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, GlobalAveragePooling1D

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(28 * 28, 1))) 
model.add(Dropout(0.3)) 
model.add(Dense(128, activation='relu'))                                      
model.add(Dense(64, activation='relu'))                   
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))                                          
model.add(Dense(32, activation='relu'))                   
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu')) # 큰사이즈 아닌 이상 4,4 까지 올라가지 않음
model.add(GlobalAveragePooling1D())                                    
model.add(Dense(10, activation='softmax')) # 이진분류로 출력

#3. 컴파일, 훈련 metrics=['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=1)
model.fit(x_train, y_train, epochs=1000, verbose=1, validation_split= 0.001, batch_size=128, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. 평가, 예측 predict 필요x acc로 판단

loss = model.evaluate(x_test, y_test)
ic('loss:', loss[0])
ic('accuracy', loss[1])
ic(f'{걸린시간}분')
# ic(loss)



'''

ic| 'loss:', loss[0]: 0.02277880162000656
ic| 'accuracy', loss[1]: 0.9912999868392944

DNN
ic| 'loss:', loss[0]: 0.1147436872124672
ic| 'accuracy', loss[1]: 0.9711999893188477
ic| f'{걸린시간}분': '0.4분'

DNN + GAP
ic| 'loss:', loss[0]: 1.84208083152771
ic| 'accuracy', loss[1]: 0.30320000648498535
ic| f'{걸린시간}분': '2.3분'

'''