from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Embedding, Flatten, Dropout, GlobalAveragePooling1D, MaxPool1D, GRU
from icecream import ic
import numpy as np
import time


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# print("뉴스기사의 최대길이:", max(len(i) for i in x_train)) # 2494
# print("뉴스기사의 평균길이:", sum(map(len, x_train)) / len(x_train)) # 238.71364

x_train = pad_sequences(x_train, maxlen=200, padding='pre') # (25000, 100)
x_test = pad_sequences(x_test, maxlen=200, padding='pre') # (25000, 100)

# ic(np.unique(x_train.shape)) # 100, 25000



ic(np.unique(x_train))

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=200))
model.add(GRU(128, activation='relu', return_sequences=True))
model.add(Conv1D(64, 2, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model.summary()

#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=256, validation_split=0.1, callbacks=[es] )
걸린시간 = round((time.time() - start) /60,1)

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)

ic(f'loss={loss[0]}')
ic(f'acc={loss[1]}')
ic(f'{걸린시간}분')

'''
ic| f'loss={loss[0]}': 'loss=0.4162569046020508'
ic| f'acc={loss[1]}': 'acc=0.8604400157928467'
ic| f'{걸린시간}분': '3.8분'
'''