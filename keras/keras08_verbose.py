from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from icecream import ic
from tensorflow.python.keras.backend import batch_dot
import matplotlib.pyplot as plt
import time


#1. 데이터
x1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]) 

x = np.transpose(x1) # (10, 3) 훈련용 데이터에서는 x의 행의 개수가 y의 행의 개수와 같아야 한다(그렇게 때문에 여기서는 transpose를 해줌).


y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) # 

ic(y.shape) # (10,)  



#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(8)),
model.add(Dense(6)),
model.add(Dense(7)),
model.add(Dense(9)),
model.add(Dense(5)),
model.add(Dense(3)),
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
model.fit(x,y, epochs=1000, batch_size=1, verbose=0) 
# verbose => (0-> 결과만 보임/1-> 학습되는 과정이 보임 /2-> 프로그레스바가 보이지 않음/ 3-> epoch만 보여줌
# 학습되는 모습이 출력되면 그 만큼 딜레이가 발생/ 훈련시키는데 소모하는 시간이 큼)
# epochs -> 1000일때 걸린시간: 0: 22.858906507492065 / 1: 30.828540325164795  / 2:24.218281507492065 / 3: 23.674671173095703
end = time.time() - start # 걸린시간, time.time() -> 현재시간
print("걸린시간: ", end)

# verbose = 1일때, batch=1,10 인경우 시간측정


#4. 평가, 예측
# print('데이터 x의 shape:', x.shape)
# loss = model.evaluate(x, y)
# print('loss값:', loss)
# x_array = np.array([[10, 1.3, 1]]) 
# print('예측값 x의 shape:', x_array.shape) # (1, 3) 열우선 행 무시 훈련값으로 준 x가 (10,3) 이기 때문에 열값만 맞으면 됨
# x_pred = model.predict(x_array)
# print('x의 예측값:', x_pred)

'''

loss값: 0.007149036042392254
예측값 x의 shape: (1, 3)
x의 예측값: [[20.113182]]

'''

y_pred = model.predict(x)
