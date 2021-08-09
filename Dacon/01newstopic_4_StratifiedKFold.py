import numpy as np
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from icecream import ic
from sklearn.metrics import accuracy_score,log_loss
from sklearn.model_selection import StratifiedKFold
import datetime

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import Adam

path = './Dacon/_data/newstopic/'
train = pd.read_csv(path + 'train_data.csv',header=0, encoding='UTF8')

test = pd.read_csv(path + 'test_data.csv',header=0, encoding='UTF8')

submission = pd.read_csv(path + 'sample_submission.csv')

topic_dict = pd.read_csv(path + 'topic_dict.csv')

# null값 제거
# datasets_train = datasets_train.dropna(axis=0)
# datasets_test = datasets_test.dropna(axis=0)

# x = datasets_train.iloc[:, -2]
# y = datasets_train.iloc[:, -1]
# x_pred = datasets_test.iloc[:, -1]
train['doc_len'] = train.title.apply(lambda words: len(words.split()))

x_train = np.array([x for x in train['title']])
x_predict = np.array([x for x in test['title']])
y_train = np.array([x for x in train['topic_idx']])

def text_cleaning(docs):
    for doc in docs:
        doc = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", doc)
    return docs
x = text_cleaning(x_train)
x_predict = text_cleaning(x_predict)
# ic(x.shape) ic| x.shape: (45654,)

# 불용어 제거, 특수문자 제거
# import string
# def define_stopwords(path):
#     sw = set()
#     for i in string.punctuation:
#         sw.add(i)

#     with open(path, encoding='utf-8') as f:
#         for word in f:
#             sw.add(word)

#     return sw
# x = define_stopwords(x)

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()  
tokenizer.fit_on_texts(x)
sequences_train = tokenizer.texts_to_sequences(x)
sequences_test = tokenizer.texts_to_sequences(x_predict)

#리스트 형태의 빈값 제거  --> 양방향에서는 오류남..
# sequences_train = list(filter(None, sequences_train))
# sequences_test = list(filter(None, sequences_test))

#길이 확인
# x1_len = max(len(i) for i in sequences_train)
# ic(x1_len) # ic| x1_len: 11
# x_pred = max(len(i) for i in sequences_test)
# ic(x_pred) # ic| x_pred: 9

xx = pad_sequences(sequences_train, padding='pre', maxlen = 14)
# ic(xx.shape) ic| xx.shape: (42477, 11)
yy = pad_sequences(sequences_test, padding='pre', maxlen=14)

y = to_categorical(y_train)

from sklearn.model_selection import train_test_split


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, Bidirectional

# model = Sequential()
# model.add(Embedding(input_dim=101082, output_dim=77, input_length=11))
# model.add(LSTM(128, activation='relu'))
# model.add(Dense(64, activation= 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation= 'relu'))
# model.add(Dense(7, activation='softmax'))
model = Sequential([Embedding(101082, 200, input_length =14),
        tf.keras.layers.Bidirectional(LSTM(units = 32, return_sequences = True, activation='relu')),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(LSTM(units = 16, return_sequences = True, activation='relu')),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(LSTM(units = 8, activation='relu')),
        Dense(7, activation='softmax')])
import datetime
import time

optimizer = Adam(learning_rate=0.0001)
model.compile(loss= 'categorical_crossentropy', optimizer= optimizer, metrics = ['acc'])

date = datetime.datetime.now()
date_time = date.strftime('%m%d_%H%M')
cvpath = './Dacon/_save/skfoldmcp/'
info = '{epoch:02d}_{val_loss:.4f}'
filepath = ''.join([cvpath, 'test', '_', date_time, '_', info, '.hdf5'])

history = model.fit(xx, y, epochs=10, batch_size=512, validation_split= 0.2)



n_fold = 5  

cv = StratifiedKFold(n_splits = n_fold, shuffle=True, random_state=66)

# 테스트데이터의 예측값 담을 곳 생성
test_y = np.zeros((yy.shape[0], 7))

# 조기 종료 옵션 추가
es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='min', baseline=None, restore_best_weights=True)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto', verbose=1, filepath=filepath)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1, patience=10, mode='auto', factor=0.1 )
start_time = time.time()
for i, (i_trn, i_val) in enumerate(cv.split(xx, y_train), 1):
    print(f'training model for CV #{i}')

    model.fit(xx[i_trn], 
            to_categorical(y_train[i_trn]),
            validation_data=(xx[i_val], to_categorical(y_train[i_val])),
            epochs=5,
            batch_size=512,
            callbacks=[es,cp])     # 조기 종료 옵션
                      
    test_y += model.predict(yy) / n_fold  

topic = []
for i in range(len(test_y)):
    topic.append(np.argmax(test_y[i]))
end_time = time.time() - start_time


submission['topic_idx'] = topic
ic(submission.shape)

date_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
submission.to_csv('./Dacon/_save/csv/predict' + date_time + '.csv', index=False)
