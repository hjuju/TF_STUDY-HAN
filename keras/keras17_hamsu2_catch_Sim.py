'''
06_R2_2를 카피
함수형으로 리폼
서머리로 확인
'''

from operator import mod
from re import A
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from icecream import ic
from sklearn.metrics import r2_score


x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

model1 = Sequential()
model1.add(Dense(20, input_dim=1))
model1.add(Dense(200)),
model1.add(Dense(150)),
model1.add(Dense(100)),
model1.add(Dense(100)),
model1.add(Dense(50)),
model1.add(Dense(30)),
model1.add(Dense(50)),
model1.add(Dense(1))

model1.summary()

# 함수형으로 변환
input1 = Input(shape=(1,))
dense1 = Dense(32, activation='selu')(input1)
dense2 = Dense(26, activation='selu')(dense1)
dense3 = Dense(18, activation='selu')(dense2)
dense4 = Dense(18, activation='selu')(dense3)
output1 = Dense(1)(dense4)

model = Model(inputs=input1, outputs=output1)

model.summary()




model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=2000, batch_size=2)

loss = model.evaluate(x,y)
ic(loss)

y_pred = model.predict(x)
ic(y_pred)

r2 = r2_score(y, y_pred)
ic(r2)

# ic| r2: 0.9999999999999186