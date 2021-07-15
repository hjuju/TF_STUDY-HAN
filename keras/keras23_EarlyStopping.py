from textwrap import dedent
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from icecream import ic
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from tensorflow.python.keras.constraints import MinMaxNorm


dataset = load_boston()


# validation을 넣었을때 향상 되는지 안되는지 확인

x = dataset.data # 506개의 데이터(집값)에 대한 13개의 특성 (506, 13)
y = dataset.target # 집 값에 대한 데이터 506개 (506,)

print(np.min(x), np.max(x)) # 0,0 711.0 numpy의 소수점 연산 빠름 / 데이터를 0~1 사이로 변환(데이터 전처리) / 각 데이터 사이의 비율은 바뀌지 않음
# 데이터 전처리 시 데이터 정확도, 성능, 처리속도등 모두 좋아짐
# 모든 데이터에서 최대값으로 나눠 줌 최소0, 최대값 1로 변환 -> minmax scaler
# 13개(컬럼값) 모두 다른 값으로 나누어 줘야함 -> 한개의 컬럼 으로만 하면 각 숫자의 비율이 달라져서 기준점이 달라짐 -> 나중에 값이 틀어짐



# x = (x - np.min(x)) / (np.max(x) - np.min(x)) # 정규화 처리식


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=70) # Train과 test의 스케일이 다름


from sklearn.preprocessing import MinMaxScaler, StandardScaler # 전처리로 이것저것 해보고, 하이퍼 파라미터 튜닝을 하며
scaler = MinMaxScaler()
#scaler = StandardScaler() # 표준정규분포로 변환 
scaler.fit(x_train) # xtrain에 대해서만 스케일러 해줌
x_train = scaler.transform(x_train) # 전체 데이터로 스케일링 하면 과적합이 될 수 있기때문에 나누어서 스케일링 해줌
x_test = scaler.transform(x_test) # 비율에 맞춰서 스케일링



ic(x.shape)
ic(y.shape)

ic(dataset.feature_names.shape)
ic(dataset.data.shape)
ic(dataset.DESCR) # 데이터셋 기술서 


model1 = Sequential()
model1.add(Dense(512, input_dim=13, activation='relu'))
model1.add(Dense(200, activation='relu'))
model1.add(Dense(100, activation='relu'))
model1.add(Dense(100, activation='relu'))
model1.add(Dense(70, activation='relu'))
model1.add(Dense(1))

model1.summary()
'''
# 함수형 모델로 변환
input1 = Input(shape=(13,))
dense1 = Dense(512, activation='relu')(input1)
dense2 = Dense(400, activation='relu')(dense1)
dense3 = Dense(300, activation='relu')(dense2)
dense4 = Dense(256, activation='relu')(dense3)
dense5 = Dense(200, activation='relu')(dense4)
dense6 = Dense(128, activation='relu')(dense5)
dense7 = Dense(100, activation='relu')(dense6)
output1 = Dense(1)(dense7)

model = Model(inputs=input1, outputs=output1)
model.summary()
'''


model1.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='loss', patience=20, mode='auto', verbose=1) # mode=max, auto (accuracy측정,)
# patience= 최소 로스값이 갱신되지 n번까지 갱신되지 않으면 끝냄(n번 내에서 다시 최저점으로 갱신되면 그때부터 n번 다시 카운트,반복), mode='min' 값이 최소면 거기부터 다시 patioence n번 
hist = model1.fit(x_train, y_train, epochs=1000, verbose=1, batch_size=8, validation_split=0.2, shuffle=True, callbacks=[es])

print(hist.history.keys()) # dict_keys(['loss', 'val_loss'])
print("="*200)
print(hist.history['loss']) # loss의 히스토리를 반환함
print("="*200)
print(hist.history['val_loss']) # val_loss의 히스토리를 반환함

import matplotlib.pyplot as plt

plt.plot(hist.history['loss']) # x= epoch, y = hist.history['loss']
plt.plot(hist.history['val_loss'])

plt.title("로스, 발로스") 
plt.xlabel('epochs')
plt.ylabel("loss, val_loss")
plt.legend(['train loss', 'val loss']) # 범례 추가(순서대로 첫번째, 두번째 매치)
plt.show()

loss = model1.evaluate(x_test, y_test)

ic(loss)
y_predict = model1.predict(x_test)

r2 = r2_score(y_test, y_predict)
ic(r2)


 
'''
StandardScaler
ic| loss: 10.260350227355957
ic| r2: 0.8940825758921485
model1.fit(x_train, y_train, epochs=300,verbose=1, batch_size=32, shuffle=True, callbacks=[es])

MinMaxScaler
ic| loss: 8.533263206481934
ic| r2: 0.911911260644259
'''