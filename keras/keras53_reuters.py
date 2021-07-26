from tensorflow.keras.datasets import reuters
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from icecream import ic
from tensorflow.python.keras import callbacks

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2) # num_words => 단어사전의 개수

# ic(x_train[0], type(x_train[0]))

# ic(y_train)  # 3

# ic(len(x_train[0]), len(x_train[1])) # len(x_train[0]): 87, len(x_train[1]): 56

# ic(x_train.shape, x_test.shape) # x_train.shape: (8982,), x_test.shape: (2246,)
# ic(y_train.shape, x_test.shape) # y_train.shape: (8982,), x_test.shape: (2246,)

# ic(type(x_train)) # type(x_train): <class 'numpy.ndarray'>


print("뉴스기사의 최대길이:", max(len(i) for i in x_train)) # 2376
print("뉴스기사의 평균길이:", sum(map(len, x_train)) / len(x_train)) # 145.5개의 평균 단어



# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

# 전처리
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping
# import time

# x_train = pad_sequences(x_train, maxlen=100, padding='pre') # (8982, 100)
# x_test = pad_sequences(x_test, maxlen=100, padding='pre') # (2246, 100)
# # ic(type(x_train), type(x_train[0])) # type(x_train): <class 'numpy.ndarray'>, type(x_train[0]): <class 'numpy.ndarray'>

# # ic(x_train[0]) # 0이 13개 생김 (maxlen에 의해 필요한만큼 0이 채워짐)

# # ic(np.unique(y_train)) # 0 ~ 45

# ic(np.unique(x_train.shape))



# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# # ic(y_train.shape, y_test.shape) # y_train.shape: (8982, 46), y_test.shape: (2246, 46)

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Conv1D, Embedding, Flatten


# #2. 모델 생성
# model = Sequential()
# model.add(Embedding(input_dim=1000000, output_dim=128 ))
# model.add(LSTM(64, activation='relu'))
# model.add(Flatten())
# model.add(Dense(46, activation='softmax'))

# # model.summary()

# #3. 컴파일, 훈련
# es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# start = time.time()
# model.fit(x_train, y_train, epochs=300, batch_size=1024, validation_split=0.2, callbacks=[es] )
# 걸린시간 = round((time.time() - start) /60,1)

# #4. 평가, 예측

# loss = model.evaluate(x_test, y_test)

# ic(f'loss={loss[0]}')
# ic(f'acc={loss[1]}')
# ic(f'{걸린시간}분')

# '''
# ic| f'loss={loss[0]}': 'loss=4043.264404296875'
# ic| f'acc={loss[1]}': 'acc=0.3566340208053589'
# ic| f'{걸린시간}분': '17.2분
# '''




