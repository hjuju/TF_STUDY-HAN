import numpy as np
from sklearn.datasets import load_breast_cancer
from icecream import ic
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 2진 분류

# 1. 데이터
datasets = load_breast_cancer()
# ic(datasets.DESCR)
# ic(datasets.feature_names)

x = datasets.data
y = datasets.target  #2진 분류(0 or 1)

# ic(x.shape, y.shape)   #(569, 30), (569,)
# ic(y[:40])  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
# ic(np.unique(y))   # [0, 1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=9)

# 1-2. x 데이터 전처리
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# ic(x_train.shape, x_test.shape)   # x_train.shape: (398, 30), x_test.shape: (171, 30)
x_train = x_train.reshape(398, 30, 1, 1)
x_test = x_test.reshape(171, 30, 1, 1)

# 1-3. y 데이터 전처리
# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# ic(y_train.shape, y_test.shape)   #  y_train.shape: (398, 2), y_test.shape: (171, 2)


# 2. 모델
model = Sequential()
model.add(Conv2D(128, kernel_size=(1,1), input_shape=(30,1,1), activation='relu'))
model.add(Conv2D(64, (1,1), activation='relu'))
model.add(Conv2D(64, (1,1), activation='relu'))
model.add(Conv2D(64, (1,1), activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation='sigmoid'))  # sigmoid : 0과 1사이의 값  # 스칼라 569인 벡터 1개


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   # binary_crossentropy : 2진 분류   # metrics(결과에 반영X):평가지표

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=5, mode='min')

print('************************')
ic(x_train.shape, x_test.shape)

import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, callbacks=[es])
end = time.time() - start


# 4. 평가, 예측
results = model.evaluate(x_test, y_test)   # 무조건 낮을수록 좋다
print('걸린시간 :', end)
print('loss :', results[0])
print('accuracy :', results[1])

ic(y_test[-5:-1])
y_predict = model.predict(x_test[-5:-1])
ic(y_predict)   # sigmoid 통과한 값

# 그래프 그리기
plt.rcParams['font.family'] = 'gulim'
plt.plot(hist.history['loss'])

plt.title('유방암')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend('유방암 loss')
plt.show()


'''
ic| loss: 0.013418859802186489
loss : 0.41947320103645325
accuracy : 0.9766082167625427
*cnn + GAP
'''