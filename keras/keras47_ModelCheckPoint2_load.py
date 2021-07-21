from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler, OneHotEncoder
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import pandas as pd
from sklearn.datasets import load_diabetes
from icecream import ic
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

# save파일명: keras46_1_save_model.h5


#1. 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

ic(x.shape, y.shape) # x.shape: (442, 10), y.shape: (442,)

# ic(datasets.feature_names)
# # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# ic(datasets.DESCR)

# ic(y[:30]) # 30개 데이터 출력

# ic(np.min(y), np.max(y)) # 최소, 최대값 

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=60)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성

# model = Sequential()
# model.add(Dense(256, input_shape=(10,)))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1))







#3. 컴파일, 훈련
#es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
#cp = ModelCheckpoint(monitor='val_loss', patience=3, mode='min', filepath='./_save/ModelCheckPoint/keras47_MCP.hdf5')
# model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# start = time.time()
# model.fit(x_train, y_train, epochs=500, verbose=2, batch_size=8, validation_split=0.2, shuffle=True, callbacks=[es, cp])
# 걸린시간 = round((time.time() - start) /60,1)

# model.save('./_save/ModelCheckPoint/keras47_model_save.h5')
# model.save_weights('./_save/keras46_1_save_weights_2.h5')

model = load_model('./_save/ModelCheckPoint/keras47_model_save.h5') # save_model ic| r2:  0.5196134587073479
#model = load_model('./_save/ModelCheckPoint/keras47_MCP.hdf5') # 체크포인트 0.5186985906800602

model.summary()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict(x_test)
# ic(y_predict)

r2 = r2_score(y_test, y_predict)
ic(r2)

# ic(걸린시간)


'''
ic| loss: [2927.165283203125, 43.270164489746094]
ic| r2: 0.52195102666388
ic| 걸린시간: 0.1

load_model
ic| loss: [3198.572265625, 44.052711486816406]
ic| r2: 0.47762625346072596


'''