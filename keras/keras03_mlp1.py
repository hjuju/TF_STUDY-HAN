from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from icecream import ic
from tensorflow.python.keras.backend import batch_dot
import matplotlib.pyplot as plt


'''
array

(1): [1, 2, 3] => 
(2): [[1, 2, 3]] =>
(3): [[1, 2],[3, 4],[5, 6]] =>
(4): [[[1, 2, 3],[4, 5, 6]]] =>
(5): [[[1, 2]],[[3, 4]],[[5, 6]]] =>
(6): [[[1],[2]], [[3],[4]], [[3],[4]]] =>

'''



#1. 데이터
x1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]) # (2, 10)
 
x = np.transpose(x1) # (10, 2) 훈련용 데이터에서 x의 행의 개수가 y의 행의 개수와 같아야 한다(그렇게 때문에 여기서는 transpose를 해줌).

ic(x.shape)
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) # 

y = np.transpose(y)

ic(y.shape) # (10,)  



#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(8)),
model.add(Dense(6)),
model.add(Dense(7)),
model.add(Dense(9)),
model.add(Dense(5)),
model.add(Dense(3)),
model.add(Dense(2))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=1)


#4. 평가, 예측
loss = model.evaluate(x, y)
ic(loss)
x_array = np.array([[10, 1.3]]) 
ic(x_array.shape) # (1, 2) 열우선 행 무시 훈련값으로 준 x가 (10,2) 이기 때문에 열값만 맞으면 됨
x_pred = model.predict(x_array)
ic(x_pred)

'''
ic| loss: 0.008565384894609451
ic| x_array.shape: (1, 2)
ic| x_pred: array([[20.009811]], dtype=float32)

'''

y_predict = model.predict(x)

plt.scatter(x[:,0],y)
plt.scatter(x[:,1],y)
plt.plot(x, y_predict)
plt.show()


