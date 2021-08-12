from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler
# 1. 로스와 R2로 평가
# MIN MAX와 스탠다드 결과들 명시

from tensorflow.keras.callbacks import EarlyStopping
from operator import mod
import numpy as np
from numpy.matrixlib.defmatrix import matrix
# import pandas as pd
from sklearn.datasets import load_breast_cancer
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalAveragePooling1D, MaxPool1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.utils import to_categorical

#1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

# ic(x.shape, y.shape) # x.shape: (506, 13), y.shape: (506,)



# ic(datasets.feature_names)
# # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# ic(datasets.DESCR)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=60) # train 309, test 133

# scaler = QuantileTransformer()
# scaler = StandardScaler()
#scaler = PowerTransformer()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

ic(x_train.shape[1], y.shape)




# 2. 모델 구성
model = Sequential()
model.add(Conv1D(128, kernel_size=2, padding='same', activation='relu', input_shape=(30,1)))  
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(MaxPool1D())
model.add(Dropout(0.2))                                      
model.add(Conv1D(128, 2, padding='same', activation='relu')) 
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(MaxPool1D())
model.add(Dropout(0.2))                        
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(Conv1D(32, 2, padding='same', activation='relu'))
model.add(GlobalAveragePooling1D())                                       
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='acc', patience=10, mode='auto', verbose=1)
start = time.time()
model.fit(x_train, y_train, epochs=1000, verbose=1, validation_split=0.2, batch_size=4, shuffle=True, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
ic(loss)

y_predict = model.predict(x_test)
# ic(y_predict)

print('loss = ', loss[0])
print('accuracy = ', loss[1])

ic(f'{걸린시간}분')

'''
CNN + Conv1D + GAP

loss =  0.3516334593296051
accuracy =  0.9473684430122375
ic| f'{걸린시간}분': '0.4분'


이진분류

train size = 0.7
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


train size = 0.6
모델 위와 동일
c| 'loss:', loss[0]: 0.1287858933210373
ic| 'accuracy', loss[1]: 0.9868420958518982

'''