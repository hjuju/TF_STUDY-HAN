from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

# train_test_split으로 만들기! 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=6, shuffle=False, random_state=60) # 전체에서 train과 test를 4:6으로 나눔
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=3, shuffle=False, random_state=60) # 6으로 나뉜 test 데이터에서 val데이터와 나눔
'''
ic| x_train: array([1, 2, 3, 4, 5, 6, 7])
ic| x_test: array([ 8,  9, 10])
ic| x_val: array([11, 12, 13])

'''

# x_train = np.array([1,2,3,4,5,6,7]) # 실질적으로 훈련, 공부하는 것 / 7개의 데이터를 
# y_train = np.array([1,2,3,4,5,6,7])
# x_test = np.array([8,9,10]) # 테스트 데이터는 오직 평가를 위해서 사용
# y_test = np.array([8,9,10])
# x_val = np.array([11,12,13]) # 머신이 공부를 하고, 문제집을 푸는 것(model.fit에 명시)
# y_val = np.array([11,12,13]) # 훈련을 할 때 검증, 반복횟수에 따라 검증 한 뒤 다시 훈련, 다시 검증 작업을 하게 되면 성능향상

ic(x_train)
ic(x_test)
ic(x_val)

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
# model.fit(x_train,y_train, epochs=1000, batch_size=1, verbose=0,validation_data=(x_val, y_val)) # validation_data=(검증에 사용할 데이터 넣어줌)

# loss = model.evaluate(x_test,y_test)
# ic(f'로스: {loss}')

# y_pred = model.predict([11])
# ic(f'예측값: {y_pred}')


# # 통상적으로 loss가 val_loss가 더 좋게 나온다 -> loss는 과적합에 더 잘 걸린다. / val_loss를 기준으로 훈련은 시켜야함
