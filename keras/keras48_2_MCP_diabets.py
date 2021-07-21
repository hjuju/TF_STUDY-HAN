from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler
import numpy as np
from sklearn.datasets import load_diabetes
from icecream import ic
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPool2D, LSTM 
from tensorflow.keras.models import Sequential, load_model



#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

ic(x.shape, y.shape) # x.shape: (506, 13), y.shape: (506,)



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


ic(np.unique(y))


# model = Sequential()
# model.add(LSTM(40, activation='relu', input_shape=(10,1), return_sequences=True))
# model.add(Conv1D(128, 2))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Flatten())
# model.add(Dense(1))

# model.summary()

#3. 컴파일, 훈련

# es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1)
# cp = ModelCheckpointcp = ModelCheckpoint(monitor='val_loss', patience=10, mode='auto', save_best_only=True, filepath='./_save/ModelCheckPoint/keras48_2_MCP_diabets.hdf5')
# model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# start = time.time()
# model.fit(x_train, y_train, epochs=1000, verbose=1, validation_split=0.2, batch_size=8, shuffle=True, callbacks=[es, cp])
# 걸린시간 = round((time.time() - start) /60,1)

# model.save('./_save/ModelCheckPoint/keras48_2_model_save_diabets.h5')

# model = load_model('./_save/ModelCheckPoint/keras48_2_model_save_diabets.h5')
model = load_model('./_save/ModelCheckPoint/keras48_2_MCP_diabets.hdf5')

#4. 평가, 예측

y_predict = model.predict(x_test)
loss = model.evaluate(x_test, y_test)
ic(loss[0])
r2 = r2_score(y_test, y_predict)
ic(r2)
# ic(f'{걸린시간}분')

'''

모델, 체크포인트 저장

ic| loss[0]: 3650.307373046875
ic| r2: 0.40385131228988624
ic| f'{걸린시간}분': '1.3분'

모델 로드

ic| loss[0]: 3650.307373046875
ic| r2: 0.40385131228988624


체크포인트 로드

ic| loss[0]: 3525.331787109375
ic| r2: 0.4242616337912083

'''




'''

모델, 체크포인트 저장


모델 로드


체크포인트 로드



'''