import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from tensorflow.python.keras.utils.np_utils import to_categorical



datasets = load_iris()

ic(datasets.DESCR)
ic(datasets.feature_names)

x = datasets.data
y = datasets.target

ic(np.shape(x))
ic(np.shape(y))

'''
원 핫 인코딩(one hot encoding) 라벨의 종류(여기선 0,1,2) 만큼 열에 특성이 생겨남 (150,) -> (150, 3)
      0 1 2
0 -> [1,0,0]
1 -> [0,1,0]
2 -> [0,0,1]

[0,1,2,1] 의 데이터 ->
[[1,0,0]
[0,1,0]
[0,0,1]
[0,1,0]]  (4,) -> (4,3)

'''


from tensorflow.keras.utils import to_categorical

y= to_categorical(y)

ic(y[:5])
ic(np.shape(y))


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=70)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

model = Sequential()

model.add(Dense(256, activation='relu', input_shape =(4,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax')) # 다중분류는 softmax활성함수 사용


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # mse는 선을 그은 것에서 선과 비교함, 이진 분류에는 binary_crossentropy
es = EarlyStopping(monitor='loss', patience=20, mode='auto', verbose=1)
hist = model.fit(x_train, y_train, epochs=1000, verbose=1, batch_size=8, validation_split=0.2, shuffle=True, callbacks=[es])

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