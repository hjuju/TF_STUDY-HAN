from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])


x_train = x[0:7]
y_train = y[:7]
x_test = x[-3:]
y_test = y[7:]

ic(x_train)
ic(y_train)
ic(x_test)
ic(y_test)



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

