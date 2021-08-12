import numpy as np
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

#1. 데이터
x = np.array([[1,2,3],[2,3,4], [3,4,5], [4,5,6],
                [5,6,7], [6,7,8], [7,8,9],[8,9,10],
                [9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])
ic(x.shape, y.shape) # x.shape: (4, 3), y.shape: (4,)

x = x.reshape(x.shape[0], x.shape[1],1 )
x_predict = x_predict.reshape(1, x_predict.shape[0], 1) # 데이터 리쉐잎 차원 변경

#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=8, activation='relu', input_shape=(3,1))) # reshape 한 차원으로 입력, RNN 모델 구성 시 맨 처음에 simpleRNN으로 input shape만 맞춰주면 됨
# 파라미터 수 ((Input + bias) * ouput) + (output^2)  ((1+1)*10) + 100 = 120
model.add(GRU(32, activation='relu', input_shape=(3,1))) # simple RNN의 4배
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam') 
model.fit(x,y,epochs=140, batch_size=1)


#4. 평가, 예측
# x_input = np.array([5,6,7]).reshape(1,3,1) # 3차원으로 변경 후 넣어줌
result = model.predict(x_predict)
ic(result) # [[8.066318]]

# 결과값이 80 근접하게 튜닝

'''
LSTM

[79.716515]

GRU
81.54892

'''