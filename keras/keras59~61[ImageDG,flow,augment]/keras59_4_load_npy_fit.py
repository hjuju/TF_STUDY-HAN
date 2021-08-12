# 모델 완성
# np.save('./_save/_npy/k59_3_train_x.npy', arr=xy_train[0][0])
# np.save('./_save/_npy/k59_3_train_y.npy', arr=xy_train[0][1])
# np.save('./_save/_npy/k59_3_test_x.npy', arr=xy_test[0][0])
# np.save('./_save/_npy/k59_3_test_y.npy', arr=xy_test[0][1])

import numpy as np
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten 

x_train = np.load('./_save/_npy/k59_3_train_x.npy')
y_train = np.load('./_save/_npy/k59_3_train_y.npy')
x_test = np.load('./_save/_npy/k59_3_test_x.npy')
y_test = np.load('./_save/_npy/k59_3_test_y.npy')

# ic(x_train.shape) # (160, 150, 150, 3)

model = Sequential()
model.add(Conv2D(128, (2,2), input_shape=(150,150,3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()
