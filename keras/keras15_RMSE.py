from logging import root
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_squared_error

x = np.array(range(100)) # 0~99 
y = np.array(range(1, 101)) # 1~100


# 트레인과 테스트 데이터 나누기
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7,shuffle=True, random_state=66 ) 
 
# x와 y가 쌍으로 움직이며 70퍼센트를 트레인으로 주고, 테스트는 나머지(명시x), 셔플True가 디폴트,
#  랜덤값 난수(섞일때마다 성능 바뀔 수 있음)


# ic(x_train)
# ic(y_train)
# ic(x_test)
# ic(y_test)
# ic(x_train.shape, y_train.shape) # (70,) (70,)
# ic(x_test.shape, y_test.shape) # (30,) (30,)



model = Sequential() 
model.add(Dense(8, input_dim=1)) 
model.add(Dense(6)),
model.add(Dense(4)),
model.add(Dense(3)),
model.add(Dense(5)),
model.add(Dense(3)),
model.add(Dense(2)),
model.add(Dense(1))

model.compile(loss='mse', optimizer='Adam')
model.fit(x_train,y_train, epochs=50, batch_size=1)

loss = model.evaluate(x_test,y_test) 
ic(loss)

y_predict = model.predict(x_test) # x의테스트 값을 넣으면 예측값이 나옴


# 평가를 할땐 데이터 훈련 시켰지만, 원래 데이터의 값이 여자 / 훈련시킨값이 남자 -> 값이 다름
# 훈련 시킨 웨이트에 x를 집어넣으면 예측값이 나오니 예측값과 원래 테스트 값을 비교
# 남자 100인데 남자가 90명이라고 맞추면 예측을 90프로 했음/ y예측값과 y테스트 값 


# 결정계수 // 원래 y값과 훈련시켜 나온 예측값이 필요함

r2 = r2_score(y_test, y_predict) # 이 스코어를 통해 이 모델이 잘 만들어졌는지 아닌지 판단

ic(r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) # np.sqrt -> mse에 루트 씌운것을 반환

rmse = RMSE(y_test, y_predict)
ic(rmse)

