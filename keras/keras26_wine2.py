import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping
from icecream import ic
import time



datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',
                        index_col=None, header=0)

# print(datasets)
# print(datasets.shape) # (4898, 12)


datasets_np = datasets.to_numpy()
ic(datasets_np)
x = datasets_np[:,0:11]

y = datasets_np[:,[-1]]

# y = np.reshape(y, (-1,1))


# ic(np.shape(x))

# ic(np.shape(y))

# ic(x)
# ic(y)



# ic(np.unique(y)) # [3., 4., 5., 6., 7., 8., 9.]

# 1. 판다스 -> 넘파이
# 2. x와 y를 분리
# 3. sklearn의 onehotencoding 사용할 것 라벨의 시작이 원래값부터 시작함 (중간에 비어있는값 안채워준다.)(여기선 0,1,2가 안나옴)
# 3. y의 라벨을 확인 np.unique(y)


onehot_encoder = OneHotEncoder() # 0 1 2 자동채움X
onehot_encoder.fit(y)
y = onehot_encoder.transform(y).toarray()



x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.99, random_state=70)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=11))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
es = EarlyStopping(monitor='loss', patience=100, mode='auto', verbose=1)
model.fit(x_train, y_train, epochs=1000, verbose=1, batch_size=8, validation_split=0.001 , callbacks=[es])
걸린시간 = (time.time() - start) /60


loss = model.evaluate(x_test, y_test)
ic('loss:', loss[0])
ic('accuracy', loss[1])
ic(걸린시간)
# ic(loss)


y_predict = model.predict(x_test)

'''
RobustScaler
ic| 'loss:', loss[0]: 2.623243570327759
ic| 'accuracy', loss[1]: 0.7346938848495483
'''