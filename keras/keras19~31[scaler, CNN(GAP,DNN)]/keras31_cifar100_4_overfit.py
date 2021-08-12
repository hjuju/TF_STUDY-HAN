'''

overfit을 극복하자

1. 전체 훈련 데이터를 많이 -> 많으면 많을 수록 과적합이 적음

2. normalizaion(정규화) 실시 (전처리에서 , 레이어 별로)

3. dropout

global average pooling (GAP) 



'''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer,MaxAbsScaler, PowerTransformer, OneHotEncoder
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPool2D, Dropout
from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from icecream import ic
import time

#1. 데이터 전처리
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3) # 2차원으로 바꿔줌



# x_train = x_train/255.
# x_test = x_test/255.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # x_train에서만 사용가능 x_train = scaler.fit(x_train), x_train = scaler.transform(x_train)를 한줄로
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32 ,32, 3) # 스케일링 후 4차원으로 원위치



# print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]
# one = OneHotEncoder() # shape를 2차원으로 잡아야함
# y_train = y_train.reshape(-1,1) # 2차원으로 변경
# y_test = y_test.reshape(-1,1)
# one.fit(y_train)
# y_train = one.transform(y_train).toarray() # (50000, 100)
# y_test = one.transform(y_test).toarray() # (10000, 100)

# to categorical -> 3,4,6,8 되어있어도 0,1,2가 자동생성(shape에 더 유연)

# 3, 4, 5 ,6, 7 이면 그대로 3,4,5,6,7(shape가 2차원이어야함)

#2. 모델링
model = Sequential()
model.add(Conv2D(filters=256, kernel_size=(2, 2), padding='valid', activation='relu', input_shape=(32, 32, 3))) 
model.add(Dropout(0.2)) # 한 layer에서 20퍼센트의 노드를 드롭아웃함
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))                   
model.add(MaxPool2D())                       

model.add(Conv2D(128, (2, 2), padding='valid', activation='relu'))                   
model.add(Dropout(0.2))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))    
model.add(MaxPool2D())                                         

model.add(Conv2D(64, (2, 2), activation='relu'))                   
model.add(Dropout(0.2))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu')) # 큰사이즈 아닌 이상 4,4 까지 올라가지 않음
model.add(MaxPool2D())    # 556개 / 나가는 데이터를 확인해서 레이의 노드 개수 구성

model.add(Flatten())                                              
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                        metrics=['acc'])
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=64, 
                                validation_split=0.25, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. evaluating, prediction
loss = model.evaluate(x_test, y_test, batch_size=128)

print('loss = ', loss[0])
print('accuracy = ', loss[1])
ic(f'{걸린시간}분')

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

plt.show


'''
loss =  3.0406737327575684
accuracy =  0.3928000032901764

batch_size=64, validation_split=0.25
loss =  5.080616474151611
accuracy =  0.33799999952316284
ic| f'{걸린시간}분': '3.5분'

모델수정 / patience=7,epochs=100, batch_size=64, validation_split=0.25
loss =  2.777371406555176
accuracy =  0.376800000667572

dropout, patience=10
loss =  2.232557535171509
accuracy =  0.41929998993873596
ic| f'{걸린시간}분': '10.2분'


'''