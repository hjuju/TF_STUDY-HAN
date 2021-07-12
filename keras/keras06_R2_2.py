from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from icecream import ic
from sklearn.metrics import r2_score

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

model = Sequential()
model.add(Dense(20, input_dim=1))
model.add(Dense(200, activation='selu')),
model.add(Dense(150, activation='selu')),
model.add(Dense(100, activation='selu')),
model.add(Dense(100)),
model.add(Dense(50)),
model.add(Dense(30)),
model.add(Dense(50)),
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=1)

loss = model.evaluate(x,y)
ic(loss)

y_pred = model.predict(x)
ic(y_pred)

r2 = r2_score(y, y_pred)
ic(r2)