import numpy as np
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

#1. 데이터
x = np.array([[1,2,3],[2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

ic(x.shape, y.shape) # x.shape: (4, 3), y.shape: (4,)

x = x.reshape(4, 3, 1)
# (batch_size, timesteps, feature) // 각 timesteps의 feature단위로 연산
# RNN은 3차원 데이터 입력해줘야 하기 때문에 3차원으로 변경

#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=8, activation='relu', input_shape=(3,1))) # reshape 한 차원으로 입력, RNN 모델 구성 시 맨 처음에 simpleRNN으로 input shape만 맞춰주면 됨
# 파라미터 수: ((Input + bias) * ouput) + (output^2)  ((1+1)*10) + 100 = 120
# model.add(LSTM(10, activation='relu', input_shape=(3,1))) # simple RNN의 4배
model.add(GRU(10, activation='relu', input_shape=(3,1)))
# 파라미터 수: 3 * (다음 노드 수^2 +  다음 노드 수 * Shape 의 feature + 다음 노드수 )  // 3 * (10^2 + 1*10 + 10) = 360 [model.add(GRU(10, activation='relu', input_shape=(3,1)))]
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam') 
model.fit(x,y,epochs=100, batch_size=1)

#4. 평가, 예측
x_input = np.array([5,6,7]).reshape(1,3,1) # 3차원으로 변경 후 넣어줌
result = model.predict(x_input)
ic(result) # [[8.066318]]