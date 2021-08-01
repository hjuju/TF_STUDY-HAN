import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from icecream import ic
import time
import datetime
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout, GlobalAveragePooling1D, Conv1D, GRU, Input, Flatten, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Data
path = './Dacon/_data/newstopic/'
train = pd.read_csv(path + 'train_data.csv',header=0)

test = pd.read_csv(path + 'test_data.csv',header=0)

submission = pd.read_csv(path + 'sample_submission.csv')

topic_dict = pd.read_csv(path + 'topic_dict.csv')

# ic(train, test)

def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", sent)
    return sent_clean

train["cleaned_title"] = train["title"].apply(lambda x : clean_text(x))
test["cleaned_title"]  = test["title"].apply(lambda x : clean_text(x))

ic(train, test)

train_text = train["cleaned_title"].tolist()
test_text = test["cleaned_title"].tolist()

# ic(train_text)

train_label = np.asarray(train.topic_idx)

tfidf = TfidfVectorizer(analyzer='word', sublinear_tf=True, ngram_range=(1, 2), max_features=150000, binary=False)

tfidf.fit(train_text)

train_tf_text = tfidf.transform(train_text).astype('float32')
test_tf_text  = tfidf.transform(test_text).astype('float32')
y_train = np.array([x for x in train['topic_idx']])
ic(train_tf_text.shape, test_tf_text.shape)
# ic(train_tf_text[:1])
ic(train_label.shape)


# Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping

input1 = Input(shape=(150000,))
emb = Embedding(input_dim=150000, output_dim=200, input_length=14)(input1)
bid1 = LSTM(512, return_sequences=True, activation='relu')(input1)
bid2 = Bidirectional(LSTM(128, return_sequences=True, activation='relu'))(bid1)
drp = Dropout(0.3)(bid2)
bid3 = Bidirectional(LSTM(256, return_sequences=True, activation='relu'))(drp)
gru1 = GRU(512, activation='relu', return_sequences=True)(bid3)
drp2 =Dropout(0.3)(gru1)
gru2 = GRU(256, activation='relu', return_sequences=True)(drp2)
flt1 = Flatten()(gru1)
ds1 = Dense(128,activation='relu')(flt1)

output11 = Dense(256, activation='relu')(ds1)
drp = Dense(0.4)(output11)
output12 = Dense(128, activation='relu')(drp)
output13 = Dense(64, activation='relu')(output12)

output21 = Dense(256, activation='relu')(ds1)
output22 = Dense(128, activation='relu')(output21)
drp = Dropout(0.3)(output22)
output23 = Dense(64, activation='relu')(drp)

merge1 = concatenate([output13, output23])
merge2 = Dense(32, activation='relu')
last_output = Dense(7, activation='softmax')(merge2)

model = Model(inputs= input1, outputs=last_output)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

import time
start_time = time.time()
model.fit(train_tf_text[:40000], train_label[:40000], epochs=7, batch_size=16, validation_data=(train_tf_text[40000:], train_label[40000:]))
# model.fit(train_tf_text, train_label, epochs=5, batch_size=128, validation_split=0.2)
duration_time = time.time() - start_time

# Predict
y_predict = model.predict(test_tf_text)
y_predict = np.argmax(y_predict, axis=1)

# Results make to_csv submissions
# ic(len(test_tf_text))
# topic = []
# for i in range(len(test_tf_text)):
#     topic.append(np.argmax(test_tf_text[i]))   # np.argmax -> 최대값의 색인 위치

submission['topic_idx'] = y_predict
ic(submission.shape)


date_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
submission.to_csv('./Dacon/_save/predict' + date_time + '.csv', index=False)