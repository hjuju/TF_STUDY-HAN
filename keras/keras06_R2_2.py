from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from icecream import ic
from sklearn.metrics import r2_score

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

model = Sequential()
model.add(Dense(12, input_dim=1))
model.add(Dense(10)),
model.add(Dense(8)),
model.add(Dense(7)),
model.add(Dense(6)),
model.add(Dense(4)),
model.add(Dense(3)),
model.add(Dense(2)),
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=500, batch_size=1)

loss = model.evaluate(x,y)
ic(loss)

y_pred = model.predict(x)
ic(y_pred)

r2 = r2_score(y, y_pred)
ic(r2)