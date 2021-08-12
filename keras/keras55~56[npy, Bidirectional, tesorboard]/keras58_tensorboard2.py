from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from icecream import ic

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,4,3,5,6,7,8,9,10])

model = Sequential() # 인공신경망 구성(인풋, 아웃풋 사이의 히든레이어 구성)
model.add(Dense(8, input_dim=1)) # 각 layer의 인풋, 아웃풋을 연결하여 설정(여기서는 인풋 한개에 아웃풋 5개)
model.add(Dense(6)) # 하이퍼 파라미터 튜닝 -> 순차적 모델이기 때문에 위의 아웃풋이 아래의 인풋이 되기 때문에 input명시 안해도 됨
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))


tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0,
                    write_graph=True, write_images=True)

model.compile(loss='mse', optimizer='Adam')
model.fit(x,y, epochs=50, batch_size=1,validation_split=0.2, callbacks=[tb])

loss = model.evaluate(x,y)
ic(loss)

x_pred = model.predict([6])
ic(x_pred)


'''
ic| loss: 0.3892192840576172
ic| x_pred: array([[5.867008]], dtype=float32)

'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from icecream import ic

x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

model = Sequential() # 인공신경망 구성(인풋, 아웃풋 사이의 히든레이어 구성)
model.add(Dense(8, input_dim=1)) # 각 layer의 인풋, 아웃풋을 연결하여 설정(여기서는 인풋 한개에 아웃풋 5개)
model.add(Dense(6)) # 하이퍼 파라미터 튜닝 -> 순차적 모델이기 때문에 위의 아웃풋이 아래의 인풋이 되기 때문에 input명시 안해도 됨
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='Adam')
model.fit(x,y, epochs=600, batch_size=1)

loss = model.evaluate(x,y)
ic(loss)

x_pred = model.predict([6])
ic(x_pred)


'''
ic| loss: 0.3892192840576172
ic| x_pred: array([[5.867008]], dtype=float32)

'''


###########################################################
'''
커맨드 창에서 텐서보드 로그파일 있는 곳까지 가기
드라이브 위치 ex) D드라이브 -> D:
폴더 ed) cd study
디렉토리 확인 dir/w
tesorboard --logdir=.
웹에서 http://127.0.0.1:6006
또는 http://localhost:6006/ 실행

'''