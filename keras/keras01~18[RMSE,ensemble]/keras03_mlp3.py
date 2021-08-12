from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from icecream import ic
from tensorflow.python.keras.backend import batch_dot
import matplotlib.pyplot as plt


#1. 데이터
x1 = np.array([range(10), range(21, 31), range(201, 211)]) # (10,3)



x = np.transpose(x1) # (3, 10) 데이터에서는 x의 행의 개수가 y의 행의 개수와 같아야 한다(그렇게 때문에 여기서는 transpose를 해줌).


y1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]) 

y = np.transpose(y1)
ic(y.shape)   


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(8)),
model.add(Dense(6)),
model.add(Dense(7)),
model.add(Dense(9)),
model.add(Dense(5)),
model.add(Dense(3)),
model.add(Dense(3))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=1)



#4. 평가, 예측
print('데이터 x 의 shape:', x.shape)
loss = model.evaluate(x, y)
print('loss값:', loss)
x_array = np.array([[0, 21, 201]]) 
print('예측값 x 의 shape:', x_array.shape)
x_pred = model.predict(x_array)
print('x 의 예측값:', x_pred)

y_pred = model.predict(x)

'''

loss값: 0.013871523551642895

x 의 예측값: [[0.92548555 1.1930591  9.946752]]

'''

plt.scatter(x,y)
plt.plot(x, y_pred)
plt.show()

