from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from icecream import ic
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

dataset = load_boston()

x = dataset.data # 506개의 데이터(집값)에 대한 13개의 특성 (506, 13)
y = dataset.target # 집 값에 대한 데이터 506개 (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=70, shuffle=True, random_state=70 )

ic(x.shape)
ic(y.shape)

ic(dataset.feature_names.shape)
ic(dataset.data.shape)
ic(dataset.DESCR) # 데이터셋 기술서 


model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1)

loss = model.evaluate(x_test, y_test)

ic(loss)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
ic(r2)



