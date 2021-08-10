from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터 and연산
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0]

#2. 모델
# model = LinearSVC()
# model = SVC() # 다층 퍼셉트론 적용
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))


#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. 평가, 예측
y_predict = model.predict(x_data)

print(x_data, '의 예측결과: ', y_predict)

results = model.evaluate(x_data, y_data)
ic(results[0])

acc = accuracy_score(y_data, y_predict)
ic(acc)