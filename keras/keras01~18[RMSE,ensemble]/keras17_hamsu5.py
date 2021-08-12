# 


from operator import mod
import numpy as np
from icecream import ic
from tensorflow.python.keras.engine.input_layer import Input

# 데이터

x = np.array([range(100), range(301, 401), range(1, 101),
                range(100), range(401, 501)])

x = np.transpose(x)

ic(x.shape) # (100, 5)

y= np.array([range(711, 811), range(101, 201)])
y = np.transpose(y)

ic(y.shape) # (100, 2)  input = 5 , output = 2 


#2 모델구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input



input1 = Input(shape=(5,))  # 레이어 이름을 연속적으로 xx와 같이 중복적으로 주어도 가능함/ 하지만 분기가 많거나 합칠 때 헷갈릴 수 있음
dense1 = Dense(3)(input1) 
dense2 = Dense(4)(dense1) 
dense3 = Dense(10)(dense2)
output1 = Dense(2)(dense3)

model = Model(inputs=input1, outputs=output1) # 함수형은 마지막에 모델 선언해줌
# 모델을 여러개 엮거나 풀어서 쓸 경우 input과 output 명시 한 것을 잡아주면 엮거나(앙상블) 풀 수 있음, 레이어를 건너뛸 수있다(Dense 붙여주는거에 따라)

model.summary() # 함수형 모델의 summary는 인풋부터 레이어를 명시해줌 / 시퀀셜 모델과 함수형 모델과 동일함, 단지 명시만 다르게 해줌


# model = Sequential()
# model.add(Dense(3, input_shape=(5,)))
# model.add(Dense(4))
# model.add(Dense(10))
# model.add(Dense(2))



#3. 컴파일, 훈련

#4. 평가, 예측