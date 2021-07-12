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
model.compile(loss='mse', optimizer='adam', metrics=['mae']) # 로스는 mse로 계산하여 훈련에 영향을 끼치지만, mae는 훈련에 영향X, 어떤 결과가 나오는지만 출력, ['mae','mse']와 같이 여러개 가능
start = time.time()
model.fit(x,y, epochs=100, batch_size=1, verbose=1) 

end = time.time() - start # 걸린시간, time.time() -> 현재시간
print("걸린시간: ", end)


# 4. 평가, 예측
print('데이터 x의 shape:', x.shape)
loss = model.evaluate(x, y)
print('loss값:', loss) # metrict를 설정하면 설정한 손실함수값 포함해서 로스의 결과 두 개가 리스트로 나옴 
x_array = np.array([[10, 1.3, 1]]) 
print('예측값 x의 shape:', x_array.shape) # (1, 3) 열우선 행 무시 훈련값으로 준 x가 (10,3) 이기 때문에 열값만 맞으면 됨
x_pred = model.predict(x_array)
print('x의 예측값:', x_pred)


y_pred = model.predict(x)

'''
#1 mae란? mean absolute error -   평균절대오차 

​AE - absolute error - 절대오차 -

        실제 값과 측정(예측) 값과의 차이 

       Δx=xi−x 
 
        xi - 측정값 / x - 실제값

#2 rmse란? - 평균제곱근오차
큰 오류값 차이에 대해서 크게 패널티를 준다

#3 rmsle란?
RMSE 계산에 Log 를 취하는 절차를 더 해서 큰 값이 계산 전체에 지나친 영향을 미치지 못하게 제어하는 것 

'''