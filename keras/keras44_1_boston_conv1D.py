from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.datasets import load_boston
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.core import Flatten



#1. 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target

# ic(x.shape, y.shape) # x.shape: (506, 13), y.shape: (506,)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=60) # train 309, test 133

# scaler = QuantileTransformer()
scaler = StandardScaler()
# scaler = PowerTransformer()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# ic(x_train.shape, x_test.shape)


# ic(np.unique(y))


model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(13,1), return_sequences=True))
model.add(Conv1D(128, 2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Flatten())
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1)
start = time.time()
model.fit(x_train, y_train, epochs=1000, verbose=1, validation_split=0.2, batch_size=32, shuffle=True, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. 평가, 예측

y_predict = model.predict(x_test)
loss = model.evaluate(x_test, y_test)
ic(loss[0])
r2 = r2_score(y_test, y_predict)
ic(r2)
ic(f'{걸린시간}분')


'''
LSTM

ic| loss[0]: 3588.6279296875
ic| r2: 0.41392449176116985
ic| f'{걸린시간}분': '6.3분'

CNN
ic| loss: [14.31032943725586, 2.9922759532928467]
ic| r2: 0.8017186882057898
ic| f'{걸린시간}분': '0.6분'

LSTM + Conv1D
ic| loss[0]: 22.978023529052734
ic| r2: 0.7260213135951661
ic| f'{걸린시간}분': '0.9분'

'''