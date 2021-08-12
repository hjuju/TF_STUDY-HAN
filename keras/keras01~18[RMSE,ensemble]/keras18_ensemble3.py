import numpy as np
from icecream import ic
from tensorflow.python.ops.gen_control_flow_ops import merge
x1 = np.array([range(100), range(301, 401), range(1, 101)])
# x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
# x2 = np.transpose(x2)
y1 = np.array(range(1001, 1101))
y2 = np.array(range(1901, 2001))


ic(x1.shape, y1.shape, y2.shape) # ic| x1.shape: (100, 3), x2.shape: (100, 3), y1.shape: (100,)

from sklearn.model_selection import train_test_split 
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, y1, y2, train_size=0.7, random_state=60)

#2. 모델 구성(앙상블 모델)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1 모델1
input1 = Input(shape=(3,))
dense1 = Dense(128, activation='relu', name='dense1')(input1)
dense2 = Dense(114, activation='relu', name='dense2')(dense1)
dense3 = Dense(96, activation='relu',name='dense3')(dense2)
output1 = Dense(96,name='output1')(dense3)



output21 = Dense(7, name='Branch1')(output1)
last_output1 = Dense(1,name='dense_output1')(output21) # 첫번째 아웃풋

output22 = Dense(8, name='Branch2')(output1)
last_output2 = Dense(1,name='dense_output2')(output22) # 두번째 아웃풋
# 분기하고 싶은 부분 output 두개로 만들어서 그 전 노드(여기선 merge)땡겨 올 수 있음



# 모델 연결
model = Model(inputs=[input1],outputs=[last_output1, last_output2]) # 두개 이상 넣어주는 것이니 리스트 형태로



model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train], [y1_train, y2_train], epochs=100, batch_size=8, verbose=1) # concatenate에서 순서가 정해짐 -> output이 지정된 순서로 트레인 데이터를 넣어줘야함

#4. 평가, 예측
results = model.evaluate([x1_test], [y1_test, y2_test])
ic(results)
print("metrics['mse']: ",results[0])
print("metrics['mae']: ",results[1])



