import numpy as np
from icecream import ic

#1 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,7,6,7,11,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#2 모델

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
# model.add(Dense(1000))
model.add(Dense(1))

#3 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

optimizer = Adam(lr=0.001) # lr = learning rate 기본값 = 0.001(하이퍼 파라미터 튜닝/ 성능에 영향을 미침)

# lr가 작을수록 epochs가 받춰줘야함 epoch가 적으면 원하는 만큼 연산이 되지않을 수 있음
'''
optimizer = Adagrad(lr=0.1)
ic| loss: 286432.1875
ic| y_pred: array([[843.4783]], dtype=float32)

optimizer = Adagrad(lr=0.01)
ic| loss: 0.7535606622695923
ic| y_pred: array([[11.843696]], dtype=float32)

optimizer = Adagrad(lr=0.001) # Default LR
ic| loss: 0.7449456453323364
ic| y_pred: array([[11.76245]], dtype=float32)

optimizer = Adamax(lr=0.1)
ic| loss: 11423.921875
ic| y_pred: array([[174.65378]], dtype=float32)

optimizer = Adamax(lr=0.01)
ic| loss: 0.6752443909645081
ic| y_pred: array([[11.3538885]], dtype=float32)

optimizer = Adamax(lr=0.001) # Default LR
ic| loss: 0.6772891283035278
ic| y_pred: array([[11.3566885]], dtype=float32)

optimizer = Adadelta(lr=0.1)
ic| loss: 1.006518006324768
ic| y_pred: array([[10.250403]], dtype=float32)

optimizer = Adadelta(lr=0.01)
ic| loss: 0.702267587184906
ic| y_pred: array([[10.982009]], dtype=float32)

optimizer = Adadelta(lr=0.001) # Default LR
ic| loss: 0.6842705011367798
ic| y_pred: array([[11.165515]], dtype=float32)

optimizer = RMSprop(lr=0.1)
ic| loss: 34357128.0
ic| y_pred: array([[-9414.0625]], dtype=float32)

optimizer = RMSprop(lr=0.01)
ic| loss: 615.7593994140625
ic| y_pred: array([[-42.234287]], dtype=float32)

optimizer = RMSprop(lr=0.001) # Default LR
ic| loss: 0.86042720079422
ic| y_pred: array([[10.492834]], dtype=float32)

optimizer = SGD(lr=0.01, momentum=0.9)

optimizer = SGD(lr=0.01)

optimizer = SGD(lr=0.001)

optimizer =  Nadam(lr=0.1)
ic| loss: 968341.6875
ic| y_pred: array([[-570.69324]], dtype=float32)
optimizer =  Nadam(lr=0.01)
ic| loss: 0.9555535316467285
ic| y_pred: array([[12.3560505]], dtype=float32)
optimizer =  Nadam(lr=0.001) # Default LR
ic| loss: 1.1903140544891357
ic| y_pred: array([[12.633434]], dtype=float32)
'''

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x,y, epochs=100, batch_size=1)

#4 평가, 예측
loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])

ic(loss)
ic(y_pred)

'''
ic| loss: 1.670130722088159e-12
ic| y_pred: array([[10.999998]], dtype=float32)

adam
ic| loss: 1.8527101278305054
ic| y_pred: array([[13.31025]], dtype=float32)

lr = 0.1
ic| loss: 190.2124786376953
ic| y_pred: array([[-16.247122]], dtype=float32)

lr = 0.01
ic| loss: 2.1388678550720215
ic| y_pred: array([[9.200362]], dtype=float32)

lr = 0.001
ic| loss: 0.6770617365837097
ic| y_pred: array([[11.424211]], dtype=float32)
'''
