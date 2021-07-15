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
scaler = StandardScaler()
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
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1)
model.fit(x_train, y_train, epochs=1000, verbose=1, batch_size=8,validation_split=0.2,  shuffle=True, callbacks=[es])


print("====================평가, 예측====================")

loss = model.evaluate(x_test, y_test) # loss와 metrics 반환
ic('loss:', loss[0])
ic('accuracy', loss[1])
# ic(loss)

print("====================예측====================")
ic(y_test[:5]) # 원래 값([1,1,1,0,1])
y_predict = model.predict(x_test[:5])
ic(y_predict) # 소프트 맥스를 통과한 값(이 중 값이 큰것이 1할당 그것과 원래값 과 비교 후 accuracy 도출)
'''
ic| y_test[:5]: array([[1., 0., 0.],
                       [0., 0., 1.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.]], dtype=float32)
ic| y_predict: array([[9.9999416e-01, 5.8345349e-06, 1.3961612e-08],
                      [2.3270395e-06, 8.5259060e-04, 9.9914503e-01],
                      [2.8313827e-03, 9.7944689e-01, 1.7721687e-02],
                      [3.1562820e-05, 9.9924064e-01, 7.2778534e-04],
                      [3.3646140e-06, 8.8853319e-04, 9.9910814e-01]], dtype=float32)
'''


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
ic| 'loss:', loss[0]: 0.04439400136470795
ic| 'accuracy', loss[1]: 0.9777777791023254

'''