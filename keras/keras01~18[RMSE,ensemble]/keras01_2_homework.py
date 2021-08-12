from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from icecream import ic

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

model = Sequential()
model.add(Dense(1, input_dim=1))

model.compile(loss='mse', optimizer='Adam')
model.fit(x,y, epochs=100, batch_size=1)

loss = model.evaluate(x,y)
ic(loss)

x_pred = model.predict([6])
ic(x_pred)

