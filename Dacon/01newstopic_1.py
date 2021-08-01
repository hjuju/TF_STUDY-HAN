
import pandas as pd
import numpy as np
from icecream import ic
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Embedding, Flatten, GlobalAveragePooling1D, Dropout, GRU, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import datetime


path = './Dacon/_data/newstopic/'
train_data = pd.read_csv(path + 'train_data.csv',header=0)

test_data = pd.read_csv(path + 'test_data.csv',header=0)

submission = pd.read_csv(path + 'sample_submission.csv')

# ic(train_data.shape, test_data.shape) # train_data.shape: (45654, 3), test_data.shape: (9131, 2)


x = train_data.iloc[:,-2] 
y = train_data.iloc[:,-1] # 45654,
x_pred = test_data.iloc[:,-1]

x = x.to_numpy()
y = y.to_numpy()
x_pred = x_pred.to_numpy()



y = to_categorical(y)


# 데이터 npy저장


np.save("./Dacon/_save/_npy/newstopic_x_data.npy", arr=x) 
np.save("./Dacon/_save/_npy/newstopic_y_data.npy", arr=y)


# ic(type(x_train), type(y_train))
# ic(x_pred) # x_pred.shape: (9131, 2)

# # ic(x.shape, y.shape) # 45654,
token = Tokenizer()
token.fit_on_texts(x)
x = token.texts_to_sequences(x)
token.fit_on_sequences(x_pred)
x_pred = token.texts_to_sequences(x_pred)



# print("뉴스기사의 최대길이:", max(len(i) for i in x)) # 13
# print("뉴스기사의 평균길이:", sum(map(len, x)) / len(x)) # 6.62


x = pad_sequences(x, maxlen=14, padding='pre')
x_pred = pad_sequences(x_pred, maxlen=14, padding='pre')
# ic(x_train.shape, x_test.shape) # (36523, 10), x_test.shape: (9131, 10)

word_size = len(token.word_index)
# ic(word_size) # 101081
# ic(len(np.unique(x))) # 45654

# ic(np.unique(y_train)) # 0, 1, 2, 3, 4, 5, 6

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8)


# ic(y_train.shape, y_test.shape)


model = Sequential()
model.add(Embedding(input_dim=101082, output_dim=200, input_length=14))
model.add(GRU(512, activation='relu', return_sequences=True))
model.add(GRU(256, activation='relu', return_sequences=True))
model.add(GRU(128, activation='relu', return_sequences=True))
model.add(GRU(64, activation='relu', return_sequences=True))
model.add(GRU(32, activation='relu', return_sequences=True))
model.add(Flatten())
model.add(Dense(7, activation='softmax'))
model.summary()

#3. 컴파일, 훈련
date = datetime.datetime.now() 
date_time = date.strftime("%m%d_%H%M") 

filepath = './Dacon/_save/ModelCheckPoint/' 
filename = '.{epoch:04d}-{val_loss:4f}.hdf5' 
modelpath = "".join([filepath, "_newstopic_1_", date_time, "_", filename])

cp = ModelCheckpoint(monitor='val_loss', patience=10, verbose=1, mode='auto', save_best_only=True,
                    filepath= modelpath)
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1, restore_best_weights=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es, cp] )
걸린시간 = round((time.time() - start) /60,1)


model.save('./Dacon/_save/ModelCheckPoint/newstopic_save_model_2.h5')
model.save_weights('./Dacon/_save/ModelCheckPoint/newstopic_save_weights_2.h5')

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_pred)

ic(f'loss={loss[0]}')
ic(f'acc={loss[1]}')
ic(y_predict)
ic(f'{걸린시간}분')


ic(len(y_predict))
topic = []
for i in range(len(y_predict)):
    topic.append(np.argmax(y_predict[i]))  

submission['topic_idx'] = topic
ic(submission.shape)


date_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
submission.to_csv('./Dacon/_save/predict' + date_time + '.csv', index=False)