import numpy as np
from numpy.core.fromnumeric import transpose
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.datasets import load_boston
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score, mean_squared_error

x_data = np.array(range(1,101))
x_predict = np.array(range(96, 106))



size = 6
size1 = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(x_data, size)

x_predict = split_x(x_predict, size1) # (6, 5)

x = dataset[:, :-1] # (95, 5)  
y = dataset[:, -1] # (95,)
x_predict = x_predict[:, :-1]







# ic(x)
# ic(y)

ic(x.shape, y.shape)
# 시계열 데이터는 x와 y를 분리를 해줘야함


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=60) # train 309, test 133

# scaler = QuantileTransformer()
scaler = StandardScaler()
# scaler = PowerTransformer()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# ic(np.unique(y))


model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(5,1)))
model.add(LSTM(64, return_sequences=True))
model.add(Conv1D(64, 2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# model.summary()

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1)
start = time.time()
model.fit(x_train, y_train, epochs=1000, verbose=1, validation_split=0.2, batch_size=32, shuffle=True, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. 평가, 예측

result = model.predict(x_test)
loss = model.evaluate(x_test, y_test)
ic(loss[0])
r2 = r2_score(y_test, result)
ic(r2)
ic(result)
def RMSE(y_test, result):
    return np.sqrt(mean_squared_error(y_test, result)) # np.sqrt -> mse에 루트 씌운것을 반환

rmse = RMSE(y_test, result)
ic(rmse)
ic(f'{걸린시간}분')

'''
ic| loss[0]: 339.824951171875
ic| r2: 0.6179649672215815
ic| rmse: 18.43434108982547

'''