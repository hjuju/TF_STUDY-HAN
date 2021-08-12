from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# sklearn 

x = np.array(range(100)) # 0~99 
y = np.array(range(1, 101)) # 1~100


# 트레인과 테스트 데이터 나누기
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7,shuffle=True, random_state=66 ) 
 
# x와 y가 쌍으로 움직이며 70퍼센트를 트레인으로 주고, 테스트는 나머지(명시x), 셔플True가 디폴트,
#  랜덤값 난수(섞일때마다 성능 바뀔 수 있음)

ic(x_train)
ic(y_train)
ic(x_test)
ic(y_test)
ic(x_train.shape, y_train.shape) # (70,) (70,)
ic(x_test.shape, y_test.shape) # (30,) (30,)



# model = Sequential() 
# model.add(Dense(8, input_dim=1)) 
# model.add(Dense(6)),
# model.add(Dense(4)),
# model.add(Dense(3)),
# model.add(Dense(5)),
# model.add(Dense(3)),
# model.add(Dense(2)),
# model.add(Dense(1))

# model.compile(loss='mse', optimizer='Adam')
# model.fit(x_train,y_train, epochs=1000, batch_size=1)

# loss = model.evaluate(x_test,y_test)
# ic(f'로스: {loss}')

# y_pred = model.predict([11])
# ic(f'예측값: {y_pred}')

# # y_predict = model.predict(x) # x데이터 전체를 집어넣으면 x데이터 전체에 대한 예측값 출력

# # plt.scatter(x,y) # x값과 y의 원래 위치에 대한 점들을 찍음
# # plt.plot(x, y_predict, color='red') # 모델이 예측한 값(직선)에 대한 플롯
# # plt.show()

