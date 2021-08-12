import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer,MaxAbsScaler, PowerTransformer, OneHotEncoder
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model, load_model  
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPool2D, Dropout
from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from icecream import ic
import time

from tensorflow.python.ops.gen_math_ops import Min

#1. 데이터 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28 * 28 * 1)
x_test = x_test.reshape(10000, 28 * 28 * 1) # 2차원으로 바꿔줌

# x_train = x_train/255.
# x_test = x_test/255.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train) # x_train에서만 사용가능 x_train = scaler.fit(x_train), x_train = scaler.transform(x_train)를 한줄로
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1) # 스케일링 후 4차원으로 원위치

#2. 모델링
# model = Sequential()
# model.add(Conv2D(filters=256, kernel_size=(2, 2), padding='valid', activation='relu', input_shape=(28, 28, 1))) 
# model.add(Dropout(0.2)) # 한 layer에서 20퍼센트의 노드를 드롭아웃함
# model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))                   
# model.add(MaxPool2D())                                                           

# model.add(Conv2D(64, (2, 2), activation='relu'))                   
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (2, 2), padding='same', activation='relu')) # 큰사이즈 아닌 이상 4,4 까지 올라가지 않음
# model.add(MaxPool2D())    # 556개 / 나가는 데이터를 확인해서 레이의 노드 개수 구성

# model.add(Flatten())                                              
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))

model = load_model('./_save/keras45_1_save_model.h5')# 모델 로드 할 때는 load_model 임포트

model.summary()

# model.save('./_save/keras45_1_save_model.h5')



#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', patience=3, mode='auto', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                        metrics=['acc'])
start = time.time()
hist = model.fit(x_train, y_train, epochs=2, batch_size=32, 
                                validation_split=0.25, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=64)

print('loss = ', loss[0])
print('accuracy = ', loss[1])
ic(f'{걸린시간}분')


'''
model // epochs=2

loss =  0.03830191493034363
accuracy =  0.9873999953269958
ic| f'{걸린시간}분': '0.6분'

load model // epochs=2
loss =  0.04832702502608299
accuracy =  0.9850000143051147
ic| f'{걸린시간}분': '0.6분'
'''


'''
# 시각화
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
