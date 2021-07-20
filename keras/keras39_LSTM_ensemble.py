import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.ops.gen_control_flow_ops import merge
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Input
from icecream import ic
import time

x1 = np.array([[1,2,3],[2,3,4], [3,4,5], [4,5,6],
                [5,6,7], [6,7,8], [7,8,9],[8,9,10],
                [9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
x2 = np.array([[10,20,30],[20,30,40], [30,40,50], [40,50,60],
                [50,60,70], [60,70,80], [70,80,90],[80,90,100],
                [90,100,110],[100,110,120],[2,3,4],[3,4,5],[4,5,6]]) # (13, 3) (13, 3)
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_predict = np.array([55,65,75])
x2_predict = np.array([65,75,85]) 

(x1, x2) = x1.reshape(x1.shape[0], x1.shape[1],1 ), x2.reshape(x2.shape[0], x2.shape[1],1 )
(x1_predict, x2_predict)  = x1_predict.reshape(1, x1_predict.shape[0], 1), x2_predict.reshape(1, x2_predict.shape[0], 1)
 

ic(x1.shape, x2.shape) # x1.shape: (13, 3, 1), x2.shape: (13, 3, 1)



# #2. 모델 구성(앙상블 모델)



#2-1 모델1
input1 = Input(shape=(3,1))
dense1 = LSTM(32, activation='relu', name='dense1')(input1)
dense2 = Dense(16, activation='relu', name='dense2')(dense1)
dense3 = Dense(8, activation='relu',name='dense3')(dense2)
dense4 = Dense(4, activation='relu',name='dense4')(dense3)
output1 = Dense(1,name='output1')(dense4)

#2-2 모델2
input2 = Input(shape=(3,1))
dense11 = LSTM(32, activation='relu', name='dense11')(input2)
dense12 = Dense(16, activation='relu', name='dense12')(dense11)
dense13 = Dense(8, activation='relu', name='dense13')(dense12)
dense14 = Dense(4, activation='relu', name='dense14')(dense13)
output2 = Dense(1, name='output2')(dense14) # concatenate할 땐 1을 안줘도 됨

from tensorflow.keras.layers import concatenate, Concatenate

# 과제
Concatenate = Concatenate()
# merge1 = Concatenate()([output1, output2])
merge1 = Concatenate([output1, output2]) # 첫번째 와 마지막 모델의 아웃풋을 병합 / merge도 layer /분기점
merge2 = Dense(8, activation='relu')(merge1)
merge3 = Dense(4, activation='relu')(merge2)
# last_output = Dense(1)(merge3) # 마지막 layer 표현 방식(함수형 모델의 마지막 아웃풋 처럼 만들어주면 됨)


last_output = Dense(1, name='dense_output')(merge3) # 첫번째 아웃풋



# 모델 연결
model = Model(inputs=[input1, input2],outputs=last_output)


# model.summary()

#3. 컴파일, 훈련
es = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.compile(loss='mse', optimizer='adam')
start = time.time()
model.fit([x1,x2],y,epochs=1000, batch_size=1, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. 평가, 예측
# x_input = np.array([5,6,7]).reshape(1,3,1) # 3차원으로 변경 후 넣어줌
result = model.predict([x1_predict, x2_predict])
ic(result) # [[8.066318]]
ic(f'{걸린시간}분')

# #4. 평가, 예측
# results = model.evaluate([x1_test], [y1_test, y2_test])
# ic(results)
# print("metrics['mse']: ",results[0])
# print("metrics['mae']: ",results[1])


'''
GRU_ensemble
ic| result: array([[85.941124]], dtype=float32)
ic| f'{걸린시간}분': '0.4분'

LSTM_ensemble
ic| result: array([[93.52454]], dtype=float32)
ic| f'{걸린시간}분': '0.2분'

ic| result: array([[94.56148]], dtype=float32)
ic| f'{걸린시간}분': '0.1분'

'''