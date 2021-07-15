import numpy as np
from numpy.lib.arraysetops import unique
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

datasets = load_breast_cancer()

# ic(datasets.DESCR)
# ic(datasets.feature_names)

x = datasets.data 
y= datasets.target 

ic(x.shape) # (569, 30)
ic(y.shape) # (569,)

ic(y[:10])
ic(np.unique(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=70)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

model = Sequential()

model.add(Dense(256, activation='relu', input_shape =(30,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', patience=20, mode='auto', verbose=1)
model.fit(x_train, y_train, epochs=1000, verbose=1, batch_size=8, validation_split=0.1, shuffle=True, callbacks=[es])

loss = model.evaluate(x_test, y_test)

ic(loss)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
ic(r2)



'''
model.add(Dense(256, activation='relu', input_shape =(30,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
es = EarlyStopping(monitor='loss', patience=20, mode='auto', verbose=1)

model.fit(x_train, y_train, epochs=1000, verbose=1, batch_size=8, validation_split=0.1, shuffle=True, callbacks=[es])

ic| loss: 0.017339855432510376
ic| r2: 0.9254799003120517

'''