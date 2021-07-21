from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler, OneHotEncoder
import numpy as np
from tensorflow.keras.datasets import cifar100
from icecream import ic
import time
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPool2D, LSTM 
from tensorflow.keras.models import Sequential, load_model

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
model.add(Conv1D(128, 2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
es = EarlyStopping(monitor='acc', patience=3, mode='auto', verbose=1)
cp = ModelCheckpoint(monitor='val_loss', patience=3, mode='auto', save_best_only=True, filepath='./_save/ModelCheckPoint/keras48_9_MCP_cifar100.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
model.fit(x_train, y_train, epochs=100, verbose=1, validation_split=0.2, batch_size=1024, shuffle=True, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

model.save('./_save/ModelCheckPoint/keras48_9_model_save_cifar100.h5')

# model.save('./_save/ModelCheckPoint/keras48_9_model_save_cifar100.h5')
# model.save('./_save/ModelCheckPoint/keras48_9_MCP_cifar100.hdf5')

#4. evaluating, prediction
loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0])
print('accuracy = ', loss[1])
ic(f'{걸린시간}분')


'''

모델, 체크포인트 저장


모델 로드


체크포인트 로드



'''