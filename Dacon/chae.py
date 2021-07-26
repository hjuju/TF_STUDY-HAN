import time
import datetime
import numpy as np
from numpy import argmax
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. Data Preprocessing
#read_csv 
'''
encoding 해결: 한자 영어 etc...
'''
datasets_train = pd.read_csv('../_data/train_data.csv', header=0)
datasets_test = pd.read_csv('../_data/test_data.csv', header=0)

# null값 제거
datasets_train = datasets_train.dropna(axis=0)
datasets_test = datasets_test.dropna(axis=0)
# print(datasets_train.shape, datasets_test.shape)    # (45654, 3) (9131, 2)

# x, y, x_pred 분류
x = datasets_train.iloc[:, -2]
y = datasets_train.iloc[:, -1]
x_pred = datasets_test.iloc[:, -1]
# print(x.head(), y.head(), x_pred.head())

# x, x_pred 토큰화 및 sequence화
token = Tokenizer()
token.fit_on_texts(x)
x = token.texts_to_sequences(x)
x_pred = token.texts_to_sequences(x_pred)
# print(x, x_pred)

# x, x_pred padding
max_len1 = max(len(i) for i in x)
avg_len1 = sum(map(len, x)) / len(x)
max_len2 = max(len(i) for i in x_pred)
avg_len2 = sum(map(len, x_pred)) / len(x_pred)
# print(max_len1, max_len2) # 13 11
# print(avg_len1, avg_len2) # 6.623954089455469 5.127696856861242
x = pad_sequences(x, padding='pre', maxlen=10)
x_pred = pad_sequences(x_pred, padding='pre', maxlen=10)
# print(x.shape, x_pred.shape) # (45654, 10) (9131, 10)
# print(np.unique(x), np.unique(x_pred)) # 0~101081 // 동일기준(x)으로 fit,sequence 했기 때문에 같음

# y to categorical
# print(np.unique(y)) # 0~6
y = to_categorical(y)
# print(np.unique(y)) # 0, 1

# x, y train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9)

#2. Modeling
input = Input((10, ))
e = Embedding(101082, 8)(input)
l = LSTM(8, activation='relu')(e)
output = Dense(7, activation='softmax')(l)

model = Model(inputs=input, outputs=output)

#3. Compiling, Training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

date = datetime.datetime.now()
date_time = date.strftime('%m%d_%H%M')
path = './_save/MCP/'
info = '{epoch:02d}_{val_loss:.4f}'
filepath = ''.join([path, 'test', '_', date_time, '_', info, '.hdf5'])
es = EarlyStopping(monitor='val_loss', restore_best_weights=False, mode='auto', verbose=1, patience=8)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', verbose=1, filepath=filepath)
start_time = time.time()
model.fit(x_train, y_train, epochs=16, batch_size=128, verbose=1, validation_split=0.01, callbacks=[es, cp])
end_time = time.time() - start_time

#4. Evaluating
loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0])
print('acc = ', loss[1])
print('time taken(s) = ', end_time)

'''
loss =  1.3240760564804077
acc =  0.72226482629776
time taken(s) =  30.92211627960205
'''

#5. Prediction
prediction = model.predict(x_pred)
prediction = np.argmax(prediction, axis=1) # to_categorical 되돌리기
# print(type.prediction) # numpy.ndarray

# 제출파일형식 맞추기
index = np.array([range(45654, 54785)])
index = np.transpose(index)
index = index.reshape(9131, )
file = np.column_stack([index, prediction])
file = pd.DataFrame(file)
file.to_csv('../_data/test.csv', header=['index', 'topic_idx'], index=False)




es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=512, validation_split=0.2, callbacks=[es] )
걸린시간 = round((time.time() - start) /60,1)