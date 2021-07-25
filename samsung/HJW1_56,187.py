import pandas as pd
from icecream import ic
import numpy as np
from pandas.core.tools.datetimes import Scalar
from tensorflow.python.keras.backend import concatenate, reshape, transpose
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Input , Conv1D, Concatenate, Flatten, Dropout
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from icecream import ic
import time
import datetime


ss = pd.read_csv('./samsung/_data/SAMSUNG.csv', header=0, nrows=2601, encoding='CP949')
sk = pd.read_csv('./samsung/_data/SK.csv', header=0,  nrows=2601, encoding='CP949')

ss = pd.DataFrame(ss)   
sk = pd.DataFrame(sk)


ss = ss[['시가','고가','저가','거래량','종가']]
sk = sk[['시가','고가','저가','거래량','종가']]


ss = ss.sort_index(ascending=False).to_numpy()
sk = sk.sort_index(ascending=False).to_numpy()
ic(ss)
ic(sk)


ic(ss.shape) # (2601, 5)
ic(sk.shape) # (2601, 5)


size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):  # 10 - 5 + 1 = 6행 // 행의 개수가 정해짐
        subset = dataset[i : (i + size), :]
        aaa.append(subset)
    return np.array(aaa)

samsung = split_x(ss, size)
sk = split_x(sk, size)

ic(samsung.shape)

x1_pred = samsung[-1,:]
x2_pred = sk[-1,:]

# ic(samsung)

x1 = samsung
x2 = sk


y = ss[4:,4]
y = y.reshape(-1,1)


x1 = x1.reshape(2597, 25)
x2 = x2.reshape(2597, 25)

ic(x1.shape) # (2597, 25)
ic(x2)

# x1_pred = samsung[-6:-size]



# ic(samsung)
ic(x1_pred) # (5, 5)
#ic(x2_pred) # (5, 5)
ic(x1_pred.shape) 
ic(y[:20])
ic(y.shape) 

ic(x1.shape, y.shape) # (2597, 25), y.shape: (2597, 1)
ic(x2.shape) # (2597, 25)


x1_pred = x1_pred.reshape(1,25)
x2_pred = x1_pred.reshape(1,25)

ic(y)
ic(y.shape) # (2597, 1)







# 데이터 전처리

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.8,shuffle=False)

scaler = StandardScaler()
x1_train = scaler.fit_transform(x1_train)
x1_test = scaler.transform(x1_test)
x2_train = scaler.fit_transform(x2_train)
x2_test = scaler.transform(x2_test)
x1_pred = scaler.transform(x1_pred)
x2_pred = scaler.transform(x2_pred)

ic(x1_train, x1_test)
ic(x2_train.shape, x2_test.shape)

x1_train = x1_train.reshape(x1_train.shape[0], 5, 5)
x1_test = x1_test.reshape(x1_test.shape[0], 5, 5)
x2_train = x2_train.reshape(x2_train.shape[0], 5, 5)
x2_test = x2_test.reshape(x2_test.shape[0], 5, 5)
x1_pred = x1_pred.reshape(x1_pred.shape[0],5,5)
x2_pred = x2_pred.reshape(x2_pred.shape[0],5,5)

ic(x1_train, x1_test, x2_train, x2_test, y, x1_pred, x2_pred)


ic(x1_train, x1_test)
ic(x2_train.shape, x2_test.shape)
ic(x1_pred, x2_pred)
# 모델링

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

# model.summary()

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

date = datetime.datetime.now() 
date_time = date.strftime("%m%d_%H%M") 

filepath = './_save/' 
filename = '.{epoch:04d}-{val_loss:4f}.hdf5' 
modelpath = "".join([filepath, "_samsung_", date_time, "_", filename])

cp = ModelCheckpoint(monitor='val_loss', patience=10, verbose=1, mode='auto', save_best_only=True,
                    filepath= modelpath)
es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', restore_best_weights=True)

start = time.time()
model.fit([x1_train, x2_train], y_train, epochs=30, batch_size=16, verbose=1, validation_split=0.1, callbacks=[es, cp])
걸린시간 = round((time.time() - start) /60,1)


model.save('./_save/samsung_save_model_2.h5')
model.save_weights('./_save/samsung_save_weights_1.h5')

loss = model.evaluate([x1_test, x2_test], [y_test, y_test])
y_predict = model.predict([x1_pred, x2_pred])


ic(loss)
ic(y_predict)
ic(f'{걸린시간}분')
