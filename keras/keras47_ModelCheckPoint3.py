from re import T
import numpy as np
from icecream import ic
from tensorflow.python.keras.saving.save import load_model
from tensorflow.python.ops.gen_control_flow_ops import merge
x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y = np.array(range(1001, 1101))


ic(x1.shape, x2.shape, y.shape) # ic| x1.shape: (100, 3), x2.shape: (100, 3), y1.shape: (100,)

from sklearn.model_selection import train_test_split 
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.7, random_state=60)

#2. 모델 구성(앙상블 모델)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1 모델1
input1 = Input(shape=(3,))
dense1 = Dense(128, activation='relu', name='dense1')(input1)
dense2 = Dense(64, activation='relu', name='dense2')(dense1)
dense3 = Dense(64, activation='relu',name='dense3')(dense2)
output1 = Dense(32,name='output1')(dense3)

#2-2 모델2
input2 = Input(shape=(3,))
dense11 = Dense(128, activation='relu', name='dense11')(input2)
dense12 = Dense(64, activation='relu', name='dense12')(dense11)
dense13 = Dense(64, activation='relu', name='dense13')(dense12)
dense14 = Dense(32, activation='relu', name='dense14')(dense13)
output2 = Dense(32, name='output2')(dense14) # concatenate할 땐 1을 안줘도 됨

from tensorflow.keras.layers import concatenate, Concatenate

# 과제

Concatenate = Concatenate()
# merge1 = Concatenate()([output1, output2])
merge1 = Concatenate([output1, output2]) # 첫번째 와 마지막 모델의 아웃풋을 병합 / merge도 layer
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)
last_output = Dense(1)(merge3) # 마지막 layer 표현 방식(함수형 모델의 마지막 아웃풋 처럼 만들어주면 됨)

# 모델 연결
model = Model(inputs=[input1,input2],outputs=last_output) # 두개 이상 넣어주는 것이니 리스트 형태로



model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from keras.callbacks import ModelCheckpoint, EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1, restore_best_weights=True)

#################################################################################
import datetime
date = datetime.datetime.now() # 현재시간
date_time = date.strftime("%m%d_%H%M") # 일자와 시간 (원하는 포맷 설정)

filepath = './_save/ModelCheckPoint/' # 저장 경로
filename = '.{epoch:04d}-{val_loss:4f}.hdf5' # epoch(4자리), val_loss(4자리 플롯?)
modelpath = "".join([filepath, "k47_", date_time, "_", filename])
#################################################################################

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                    # filepath='./_save/ModelCheckPoint/keras49_mcp.h5'
                    filepath= modelpath)

model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=8, 
            validation_split=0.2, verbose=1, callbacks=[es,mcp]) # concatenate에서 순서가 정해짐 -> output이 지정된 순서로 트레인 데이터를 넣어줘야함

model.save('./_save/ModelCheckPoint/keras49_model_save.h5')

from sklearn.metrics import r2_score

print('======================== 기본출력 ========================')
#4. 평가, 예측
results = model.evaluate([x1_test, x2_test], y_test)

y_predict = model.predict([x1_test, x2_test])

r2 = r2_score(y_test, y_predict)

print('loss: ',results[0])

ic(r2)
print('========================1. load_model ========================')

model2 = load_model('./_save/ModelCheckPoint/keras49_model_save.h5')
results = model2.evaluate([x1_test, x2_test], y_test)

y_predict = model2.predict([x1_test, x2_test])

r2 = r2_score(y_test,y_predict)

print('loss: ',results[0])

ic(r2)

print('========================3. Model Check Point ========================')

# model3 = load_model('./_save/ModelCheckPoint/keras49_mcp.h5')
# results = model3.evaluate([x1_test, x2_test], y_test)

# y_predict = model3.predict([x1_test, x2_test])

# r2 = r2_score(y_test,y_predict)

# print('loss: ',results[0])

# ic(r2)

'''
restore_best_weights=False
======================== 기본출력 ========================
1/1 [==============================] - 0s 14ms/step - loss: 0.0024 - mae: 0.0365
loss:  0.002388396067544818
ic| r2: 0.9999971808207698
========================1. load_model ========================
1/1 [==============================] - 0s 93ms/step - loss: 0.0024 - mae: 0.0365
loss:  0.002388396067544818
ic| r2: 0.9999971808207698
========================3. Model Check Point ========================
1/1 [==============================] - 0s 93ms/step - loss: 0.0015 - mae: 0.0308
loss:  0.0015239386120811105
ic| r2: 0.9999982011961218

저
restore_best_weights=True 브레이크 지점의 W를 장

'''