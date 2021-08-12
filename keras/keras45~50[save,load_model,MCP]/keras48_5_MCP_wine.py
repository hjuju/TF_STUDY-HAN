from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from icecream import ic
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPool2D, LSTM 
from tensorflow.keras.models import Sequential, load_model



#1. 데이터
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';', index_col=None, header=0 )

# ic(datasets)

x = datasets.iloc[:,0:11]
y = datasets.iloc[:,[-1]]

# ic(x, y)
# ic(x.shape, y.shape) 

ic(np.unique(y))

y = OneHotEncoder().fit_transform(y).toarray()


# ic(y)



x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=60) # train 309, test 133

# scaler = QuantileTransformer()
# scaler = StandardScaler()
# scaler = PowerTransformer()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

ic(x_train.shape, x_test.shape)


# model = Sequential()
# model.add(LSTM(40, activation='relu', input_shape=(11,1), return_sequences=True))
# model.add(Conv1D(128, 2))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Flatten())
# model.add(Dense(7, activation='softmax'))

# model.summary()

# #3. 컴파일, 훈련
# es = EarlyStopping(monitor='acc', patience=20, mode='auto', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss', patience=10, mode='auto', save_best_only=True, filepath='./_save/ModelCheckPoint/keras48_5_MCP_wine.hdf5')
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# start = time.time()
# model.fit(x_train, y_train, epochs=1000, verbose=1, validation_split=0.2, batch_size=32, shuffle=True, callbacks=[es, cp])
# 걸린시간 = round((time.time() - start) /60,1)

# model.save('./_save/ModelCheckPoint/keras48_5_model_save_wine.h5')

# model = load_model('./_save/ModelCheckPoint/keras48_5_model_save_wine.h5')
model = load_model('./_save/ModelCheckPoint/keras48_5_MCP_wine.hdf5')

#4. 평가, 예측

y_predict = model.predict(x_test)
loss = model.evaluate(x_test, y_test)
ic(loss[0])
ic(loss[1])
# ic(f'{걸린시간}분')

'''

모델, 체크포인트 저장

ic| loss[0]: 1.8885363340377808
ic| loss[1]: 0.5693877339363098
ic| f'{걸린시간}분': '16.9분'

모델 로드

ic| loss[0]: 1.8885363340377808
ic| loss[1]: 0.5693877339363098


체크포인트 로드

ic| loss[0]: 1.0871055126190186
ic| loss[1]: 0.557823121547699



'''