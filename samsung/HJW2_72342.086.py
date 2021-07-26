import pandas as pd
from icecream import ic
import numpy as np
from pandas.core.tools.datetimes import Scalar
from tensorflow.python.keras.backend import concatenate, reshape, transpose
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Input , Conv1D, Concatenate, Flatten, Dropout
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from icecream import ic
import time
import datetime


ss = pd.read_csv('./samsung/_data/SAMSUNG.csv', header=0,  nrows=2601, encoding='CP949')
sk = pd.read_csv('./samsung/_data/SK.csv', header=0,  nrows=2601, encoding='CP949')

ss = ss[['고가','저가','거래량','종가', '시가']]   
sk = sk[['고가','저가','거래량','종가', '시가']]

# ic(ss, sk) # 고가 저가 거래량 종가 시가
# ic(ss.shape, sk.shape) # ss.shape: (2601, 5), sk.shape: (2601, 5)


# 오름차순으로 정렬, 배열로 변경
ss = ss.sort_index(ascending=False).to_numpy()
sk = sk.sort_index(ascending=False).to_numpy()

# ic(ss, sk)

size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

split_samsung = split_x(ss, size)
split_sk = split_x(sk, size)

# ic(split_samsung, split_sk)
# ic(split_samsung.shape, split_sk.shape) # split_samsung.shape: (2597, 5, 5), split_sk.shape: (2597, 5, 5)


# ic(x1_pred.shape, x2_pred.shape) # x1_pred.shape: (5, 5, 5), x2_pred.shape: (5, 5, 5)


# ic(split_samsung.shape, split_sk.shape)      
# split_samsung.shape: (2597, 5, 5), split_sk.shape: (2597, 5, 5)

x1 = split_samsung[:-1,:,:]
x2 = split_sk[:-1,:,:]
# ic(x1.shape, x2.shape)              
# x1.shape: (2596, 5, 5),  x_2.shape: (2596, 5, 5)


x1_pred = split_samsung[-1, :]
x2_pred = split_sk[-1, :]
# ic(x1_pred.shape)    # x1_pred.shape: (5, 5)

y = ss[5:, 0]    
# ic(y.shape)     
#  y.shape: (2596,)


x1 = x1.reshape(2596,25)
x2 = x2.reshape(2596,25)
x1_pred = x1_pred.reshape(1,25)
x2_pred = x2_pred.reshape(1,25)
y = y.reshape(-1,1)


x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.8, shuffle=False)


# 데이터 전처리

scaler = StandardScaler()
x1_train = scaler.fit_transform(x1_train)
x1_test = scaler.transform(x2_test)
x2_train = scaler.fit_transform(x2_train)
x2_test = scaler.transform(x2_test)
x1_pred = scaler.transform(x1_pred)
x2_pred = scaler.transform(x2_pred)


# ic(x1_train, x1_test)
# ic(x2_train.shape, x2_test.shape)

x1_train = x1_train.reshape(x1_train.shape[0], 5, 5)
x1_test = x1_test.reshape(x1_test.shape[0], 5, 5)
x2_train = x2_train.reshape(x2_train.shape[0], 5, 5)
x2_test = x2_test.reshape(x2_test.shape[0], 5, 5)
x1_pred = x1_pred.reshape(x1_pred.shape[0],5,5)
x2_pred = x2_pred.reshape(x2_pred.shape[0],5,5)

# ic(x1_train, x1_test, x2_train, x2_test, y, x1_pred, x2_pred)


# ic(x1_train, x1_test)
# ic(x2_train.shape, x2_test.shape)
# ic(x1_pred, x2_pred)


# 모델 1

input1 = Input(shape=(5,5))
x11 = LSTM(128, return_sequences=True, activation='relu')(input1)
x12 = Conv1D(64, 2, activation='relu')(x11)
x13 = Flatten()(x12)
x14 = Dense(64, activation='relu')(x13)
x15 = Dense(32, activation='relu')(x14)
output1 = Dense(1, activation='relu')(x15)


# 모델 2

input2 = Input(shape=(5,5))
x21 = LSTM(128, return_sequences=True, activation='relu')(input2)
x22 = Conv1D(64, 2, activation='relu')(x21) 
x23 = Flatten()(x22)
x24 = Dense(64, activation='relu')(x23)
x25 = Dense(32, activation='relu')(x24)
output2 = Dense(1, activation='relu')(x25)

merge = Concatenate(axis=1)([output1, output2])
merge1 = Dense(16, activation='relu')(merge)
last_output = Dense(1)(merge1)

model = Model(inputs=[input1, input2], outputs=last_output)


# 3. 컴파일(ES), 훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1, restore_best_weights=True)


# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

date = datetime.datetime.now() 
date_time = date.strftime("%m%d_%H%M") 

filepath = './_save/' 
filename = '.{epoch:04d}-{val_loss:4f}.hdf5' 
modelpath = "".join([filepath, "_samsung2_", date_time, "_", filename])

cp = ModelCheckpoint(monitor='val_loss', patience=10, verbose=1, mode='auto', save_best_only=True,
                    filepath= modelpath)
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', restore_best_weights=True)

start = time.time()
model.fit([x1_train, x2_train], y_train, epochs=30, batch_size=16, verbose=1, validation_split=0.1, callbacks=[es, cp])
걸린시간 = round((time.time() - start) /60,1)


model.save('./_save/samsung2_save_model_2.h5')
model.save_weights('./_save/samsung2_save_weights_2.h5')

# model = load_model('./_save/_samsung2_0725_2324-1923596.hdf5')



# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], [y_test,y_test])

y_predict = model.predict([x1_pred, x2_pred])


ic(loss)
ic(y_predict)
ic(f'{걸린시간}분')

'''
ic| loss: 36774328.0
ic| y_predict: array([[72342.086]], dtype=float32)

'''