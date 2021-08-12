from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.datasets import load_iris
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score
from tensorflow.python.keras.layers.pooling import MaxPool1D



#1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target

ic(x, y) # ic| x.shape: (569, 30), y.shape: (569,)

y = to_categorical(y)

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


model = Sequential()
model.add(LSTM(40, activation='relu', input_shape=(4,1), return_sequences=True))
model.add(Conv1D(128, 2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

model.summary()

#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='acc', patience=20, mode='auto', verbose=1)
start = time.time()
model.fit(x_train, y_train, epochs=1000, verbose=1, validation_split=0.2, batch_size=32, shuffle=True, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. 평가, 예측

y_predict = model.predict(x_test)
loss = model.evaluate(x_test, y_test)
ic(loss[0])
ic(loss[1])
ic(f'{걸린시간}분')

'''
CNN + Conv1D + GAP

loss =  0.10657637566328049
accuracy =  0.9333333373069763
ic| f'{걸린시간}분': '0.2분'

DNN
ic| 'loss:', loss[0]: 0.04439400136470795
ic| 'accuracy', loss[1]: 0.9777777791023254

LSTM + Conv1D
ic| loss[0]: 0.09529005736112595
ic| loss[1]: 0.9333333373069763
ic| f'{걸린시간}분': '0.1분'
'''