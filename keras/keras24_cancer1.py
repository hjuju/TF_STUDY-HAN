import numpy as np
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
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 출력을 0과 1 사이로 한정지어줌(이진 분류에 대한 모델 완성)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # mse는 선을 그은 것에서 선과 비교함, 이진 분류에는 binary_crossentropy
es = EarlyStopping(monitor='loss', patience=30, mode='auto', verbose=1)
hist = model.fit(x_train, y_train, epochs=1000, verbose=1, batch_size=8, validation_split=0.1, shuffle=True, callbacks=[es])

print(hist.history.keys()) # dict_keys(['loss', 'val_loss'])
print("="*200)
print(hist.history['loss']) # loss의 히스토리를 반환함
print("="*200)
print(hist.history['val_loss']) # val_loss의 히스토리를 반환함


loss = model.evaluate(x_test, y_test) # loss와 metrics 반환
ic('loss:', loss[0])
ic('accuracy', loss[1])
# ic(loss)
# y_predict = model.predict(x_test)

# r2 = r2_score(y_test, y_predict)
# ic(r2)

# import matplotlib.pyplot as plt

# plt.plot(hist.history['loss']) # x= epoch, y = hist.history['loss']
# plt.plot(hist.history['val_loss'])

# plt.title("로스, 발로스") 
# plt.xlabel('epochs')
# plt.ylabel("loss, val_loss")
# plt.legend(['train loss', 'val loss']) # 범례 추가(순서대로 첫번째, 두번째 매치)
# plt.show()





'''
이진분류
model.add(Dense(256, activation='relu', input_shape =(30,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
es = EarlyStopping(monitor='loss', patience=20, mode='auto', verbose=1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='loss', patience=30, mode='auto', verbose=1)
hist = model.fit(x_train, y_train, epochs=1000, verbose=1, batch_size=8, validation_split=0.1, shuffle=True, callbacks=[es])

ic| 'loss:', loss[0]: 0.08795139938592911
ic| 'accuracy', loss[1]: 0.9766082167625427



'''