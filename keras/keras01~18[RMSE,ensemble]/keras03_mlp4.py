from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from icecream import ic
from tensorflow.python.keras.backend import batch_dot
import matplotlib.pyplot as plt

#1. 데이터
x1 = np.array([range(10)])



x = np.transpose(x1) # (10, 1) 데이터에서는 x의 행의 개수가 y의 행의 개수와 같아야 한다(그렇게 때문에 여기서는 transpose를 해줌).


y1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]) 

y = np.transpose(y1) # (10, 3)



#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(8)),
model.add(Dense(6)),
model.add(Dense(7)),
model.add(Dense(9)),
model.add(Dense(5)),
model.add(Dense(3)),
model.add(Dense(3))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=1) # 



#4. 평가, 예측


print('데이터 x shape:', x.shape)
print('데이터 y shape:', y.shape) 
loss = model.evaluate(x, y)
print('loss값:', loss)
x_array = np.array([[9]]) # predict 에는 x를 넣어 y를 예측
print('예측값 x shape:', x_array.shape)
x_pred = model.predict(x_array)
print('x 의 예측값:', x_pred)

'''

loss값: 0.008081614039838314

x 의 예측값: [[9.9896755 1.5081544 1.0552806]]

'''


y_pred = model.predict(x)
plt.scatter(x, y[:,0])
plt.scatter(x, y[:,1])
plt.scatter(x, y[:,2])
plt.plot(x, y_pred)
plt.show()


