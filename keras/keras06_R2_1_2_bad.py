from operator import mod
from re import X
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
'''

[과제 1]
# 강제로 나쁜 모델 만들어 보기
1. R2를 음수가 아닌 0.5 이하로 만들어라.
2. 데이터 건들이지 말 것
3. 레이어는 인풋 아웃풋 포함 6개 이상(히든 포함 4개 이상)
4. batch_size = 1
5. epochs = 100이상
6. 히든레이어의 노드는 10개 이상 1000개 이하
7. train 70%

[과제 2]
# keras06_R2_2의 R2를 0.9 이상으로 만들기
단톡방에 올라온 것 보다 높아지면 올리기(일요일 밤 12시까지)

[과제 3]
keras07_boston 완료하기

'''

x = np.array(range(100)) # 0~99 
y = np.array(range(1, 101)) # 1~100

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=10)


model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10)),
model.add(Dense(10)),
model.add(Dense(10)),
model.add(Dense(10)),
model.add(Dense(1))

model.compile(loss='kld', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict(x_test)


r2 = r2_score(y_test, y_predict)
ic(r2)


'''
loss='mae', optimizer='adam'
ic| loss: 0.9858219027519226
ic| r2: 0.9994628568530296

loss='msle', optimizer='adam'
ic| loss: 14.661601066589355
ic| r2: -4.3184229990491065

loss='kld', optimizer='adam'
ic| loss: 0.0
ic| r2: 0.36424262018679165


'''
