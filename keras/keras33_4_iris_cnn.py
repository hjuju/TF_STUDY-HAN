from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, OneHotEncoder
# 1. 로스와 R2로 평가
# MIN MAX와 스탠다드 결과들 명시

from tensorflow.keras.callbacks import EarlyStopping
from operator import mod
import numpy as np
from numpy.matrixlib.defmatrix import matrix
# import pandas as pd
from sklearn.datasets import load_iris
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalAveragePooling1D, MaxPool1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.utils import to_categorical

#1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target

# ic(x.shape, y.shape) # x.shape: (506, 13), y.shape: (506,)



# ic(datasets.feature_names)
# # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# ic(datasets.DESCR)
y = to_categorical(y)
ic(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=60) # train 309, test 133



# scaler = QuantileTransformer()
# scaler = StandardScaler()
#scaler = PowerTransformer()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)



ic(x_train.shape[0], x_test.shape[1])




# 2. 모델 구성
model = Sequential()
model.add(Conv1D(128, kernel_size=2, padding='same', activation='relu', input_shape=(4,1)))  
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
model.add(Dense(3, activation='softmax'))


#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='acc', patience=30, mode='auto', verbose=1)
start = time.time()
model.fit(x_train, y_train, epochs=1000, verbose=1, validation_split=0.2, batch_size=4, shuffle=True, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)


y_predict = model.predict(x_test)
# ic(y_predict)

print('loss = ', loss[0])
print('accuracy = ', loss[1])

ic(f'{걸린시간}분')

'''
CNN + Conv1D + GAP


loss =  0.10657637566328049
accuracy =  0.9333333373069763
ic| f'{걸린시간}분': '0.2분'

DNN
ic| 'loss:', loss[0]: 0.04439400136470795
ic| 'accuracy', loss[1]: 0.9777777791023254
'''