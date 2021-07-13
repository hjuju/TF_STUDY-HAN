# 보스턴 함수형으로 구현
# 서머리 확인


from textwrap import dedent
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, InputLayer
from icecream import ic
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.engine.input_layer import Input

dataset = load_boston()


# validation을 넣었을때 향상 되는지 안되는지 확인

x = dataset.data # 506개의 데이터(집값)에 대한 13개의 특성 (506, 13)
y = dataset.target # 집 값에 대한 데이터 506개 (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=70)

ic(x.shape)
ic(y.shape)

ic(dataset.feature_names.shape)
ic(dataset.data.shape)
ic(dataset.DESCR) # 데이터셋 기술서 


model1 = Sequential()
model1.add(Dense(10, input_dim=13))
model1.add(Dense(20))
model1.add(Dense(30))
model1.add(Dense(20))
model1.add(Dense(20))
model1.add(Dense(10))
model1.add(Dense(20))
model1.add(Dense(10))
model1.add(Dense(1))

model1.summary()
# 함수형 모델로 변환
input1 = Input(shape=(5,))
dense1 = Dense(10)(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(20)(dense2)
dense4 = Dense(20)(dense3)
dense5 = Dense(10)(dense4)
dense6 = Dense(20)(dense5)
dense7 = Dense(10)(dense6)
output1 = Dense(1)(dense7)

model = Model(inputs=input1, outputs=output1)
model.summary()



# model.compile(loss='mse', optimizer='adam')
# model.fit(x_train, y_train, epochs=100,verbose=3, batch_size=1, validation_split=0.3, shuffle=True)

# loss = model.evaluate(x_test, y_test)

# ic(loss)
# y_predict = model.predict(x_test)

# r2 = r2_score(y_test, y_predict)
# ic(r2)