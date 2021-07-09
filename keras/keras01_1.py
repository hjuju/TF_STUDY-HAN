from operator import mod
from re import X
from tensorflow.keras.models import Sequential # 순차적(Sequential) 모델 임포트
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.saving.model_config import model_from_config # 
import numpy as np

#1. 데이터
X = np.array([1,2,3])
Y = np.array([1,2,3]) # (정제된 데이터)

#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1)) # Y=WX+b를 표현 / W를 구하기 위한 모델 Dense(Y에대한 차원, X에 대한 차원)

#3. 컴파일과 훈련
model.compile(loss='mse', optimizer='adam') # 과제: Loss값을 정하는 mse에 대해 조사(로스를 줄이는데, 어떻게 줄이는지)

model.fit(X, Y, epochs=1300, batch_size=1) # => 하이퍼 파라미터 튜닝 // 어떤 데이터로 훈련시킬지 명시, 훈련을 시킴 / 반복횟수(전체 훈련량) / 훈련 데이터수의 간격 => W와 b가 model에 저장 됨
# 훈련할 때 마다 W가 바뀌고, W를 저장해야함

#4. 평가와 예측
loss = model.evaluate(X, Y) # mse가 반환 됨 / 평가데이터는 달라짐
print('loss:', loss)

result = model.predict([4]) # 모델 만들때 input_dim에 주어진 차수랑 맞아야함
print('4의 예측값:', result)

