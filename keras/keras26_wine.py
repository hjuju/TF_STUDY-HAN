import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from icecream import ic

# acc 0.8 이상 만들 것!!

datasets = load_wine()

x = datasets.data
y = datasets.target

print(np.shape(x))
print(np.shape(y))

y = to_categorical(y)
ic(np.shape(y))

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=70)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(13,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1)
model.fit(x_train, y_train, epochs=1000, verbose=1, batch_size=8, validation_split=0.2, callbacks=[es])


loss = model.evaluate(x_test, y_test)
ic('loss:', loss[0])
ic('accuracy', loss[1])
# ic(loss)


y_predict = model.predict(x_test)


# ic| 'loss:', loss[0]: 0.25041162967681885
# ic| 'accuracy', loss[1]: 0.9722222089767456