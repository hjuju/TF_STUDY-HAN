import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from tensorflow.keras.datasets import mnist
from icecream import ic
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ic(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) 흑백데이터 명시x
# ic(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

model = Sequential()
model.add(Dense(units=100, activation='relu', input_shape=(28,28)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax')) 

# 전처리


ic(np.unique(y_train)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ic(y_train.shape)            # (60000,1)
ic(y_test.shape) # (10000, 1)
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()


#3. 컴파일, 훈련 metrics=['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(monitor='loss', patience=5, mode='auto', verbose=1)
model.fit(x_train, y_train, epochs=1000, verbose=1, validation_split= 0.001, batch_size=128, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. 평가, 예측 predict 필요x acc로 판단

loss = model.evaluate(x_test, y_test)
ic('loss:', loss[0])
ic('accuracy', loss[1])
ic(f'{걸린시간}분')
# ic(loss)



'''
ic| 'loss:', loss[0]: 0.02277880162000656
ic| 'accuracy', loss[1]: 0.9912999868392944

DNN
ic| 'loss:', loss[0]: 0.3002414405345917
ic| 'accuracy', loss[1]: 0.9715999960899353
ic| f'{걸린시간}분': '1.4분'

'''
