import pandas as pd
import numpy as np
from icecream import ic
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
import time
import datetime

# Data
path = './Dacon/_data/newstopic/'
train = pd.read_csv(path + 'train_data.csv',header=0)

test = pd.read_csv(path + 'test_data.csv',header=0)

submission = pd.read_csv(path + 'sample_submission.csv')

# ic(train, test, submission)
# ic(train.shape, test.shape) # (45654, 3) (9131, 2)

train['doc_len'] = train.title.apply(lambda words: len(words.split()))

x_train = np.array([x for x in train['title']])
x_test = np.array([x for x in test['title']])
y_train = np.array([x for x in train['topic_idx']])

# ic(x_train, x_test, y_train)
# ic(x_train.shape, x_test.shape, y_train.shape)  # (45654,) (9131,) (45654,)

# print("뉴스기사의 최대길이:", max(len(i) for i in x_train)) # 44
# print("뉴스기사의 평균길이:", sum(map(len, x)) / len(x_train)) # 27.33

# Preprocessing

token = Tokenizer(num_words=2000)
token.fit_on_texts(x_train) 
sequences_train = token.texts_to_sequences(x_train)    
sequences_test = token.texts_to_sequences(x_test) 


x_train = pad_sequences(sequences_train, padding='pre', maxlen=14)
x_test = pad_sequences(sequences_test, padding='pre', maxlen=14)

#ic(x_train.shape, x_test.shape) # (45654, 14) (9131, 14)

y_train = to_categorical(y_train)
# ic(y_train)
# ic(y_train.shape)   # (45654, 7)


# # Modeling

model = Sequential()
model.add(Embedding(input_dim=2000, output_dim=200, input_length=14))
model.add(Bidirectional(LSTM(64, return_sequences=True, activation='relu')))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(32, return_sequences=True, activation='relu')))
model.add(Bidirectional(LSTM(16, activation='relu')))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(7, activation='softmax'))

date = datetime.datetime.now() 
date_time = date.strftime("%m%d_%H%M") 

filepath = './Dacon/_save/ModelCheckPoint/' 
filename = '.{epoch:04d}-{val_loss:4f}.hdf5' 
modelpath = "".join([filepath, "_newstopic_3_", date_time, "_", filename])

cp = ModelCheckpoint(monitor='val_loss', patience=10, verbose=1, mode='auto', save_best_only=True,
                    filepath= modelpath)
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[es, cp] )
걸린시간 = round((time.time() - start) /60,1)

y_predict = model.predict(x_test)
ic(y_predict)


ic(len(y_predict))
topic = []
for i in range(len(y_predict)):
    topic.append(np.argmax(y_predict[i]))  

submission['topic_idx'] = topic
ic(submission.shape)


date_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
submission.to_csv('./Dacon/_save/predict' + date_time + '.csv', index=False)