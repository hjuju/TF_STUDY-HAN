import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic

# Data
PATH = './Dacon/_data/newstopic/'
train = pd.read_csv(PATH + 'train_data.csv')
test = pd.read_csv(PATH + 'test_data.csv')
submission = pd.read_csv(PATH + 'sample_submission.csv')

ic(train, test, submission)
ic(train.shape, test.shape) # (45654, 3) (9131, 2)

train['doc_len'] = train.title.apply(lambda words: len(words.split()))
ic(train['doc_len'].max())

x_train = np.array([x for x in train['title']])
x_test = np.array([x for x in test['title']])
y_train = np.array([x for x in train['topic_idx']])

ic(x_train, x_test, y_train)
ic(x_train.shape, x_test.shape, y_train.shape)  # (45654,) (9131,) (45654,)

print("Article's Max Length: ", max(len(i) for i in x_train))   # 44
print("Article's Avg Length: ", sum(map(len, x_train)) / len(x_train))  # 27.33

# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()


# Preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
# token = Tokenizer(num_words=2000)
token = Tokenizer()
token.fit_on_texts(x_train)
seq_train = token.texts_to_sequences(x_train)
seq_test = token.texts_to_sequences(x_test)

print(len(seq_train), len(seq_test))
ic(seq_train[:10])
ic(np.unique(seq_train))

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(seq_train, padding='post', maxlen=14)
x_test = pad_sequences(seq_test, padding='post', maxlen=14)

ic(x_train.shape, x_test.shape) # (45654, 14) (9131, 14)

y_train = to_categorical(y_train)
ic(y_train)
ic(y_train.shape)   # (45654, 7)


# Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Embedding(2000, 200, input_length=14))
model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add(Bidirectional(LSTM(units=32)))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[es])

y_predict = model.predict(x_test)
ic(y_predict)


# Results make to_csv submissions
ic(len(y_predict))
topic = []
for i in range(len(y_predict)):
    topic.append(np.argmax(y_predict[i]))   # np.argmax -> 최대값의 색인 위치

submission['topic_idx'] = topic
ic(submission.shape)

import datetime
date_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
submission.to_csv(PATH + 'predict' + date_time + '.csv', index=False)