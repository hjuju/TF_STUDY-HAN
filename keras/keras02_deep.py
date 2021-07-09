from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from icecream import ic

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

model = Sequential() # 인공신경망 구성(인풋, 아웃풋 사이의 히든레이어 구성)
model.add(Dense(8, input_dim=1)) # 각 layer의 인풋, 아웃풋을 연결하여 설정(여기서는 인풋 한개에 아웃풋 5개)
model.add(Dense(6)) # 하이퍼 파라미터 튜닝 -> 순차적 모델이기 때문에 위의 아웃풋이 아래의 인풋이 되기 때문에 input명시 안해도 됨
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='Adam')
model.fit(x,y, epochs=600, batch_size=1)

loss = model.evaluate(x,y)
ic(loss)

x_pred = model.predict([6])
ic(x_pred)


'''
ic| loss: 0.3892192840576172
ic| x_pred: array([[5.867008]], dtype=float32)

'''