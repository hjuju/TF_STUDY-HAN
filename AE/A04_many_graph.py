# 중요하지 않은 특성들을 도태시킴 / 특징이 강한 것을 더 강하게 해주는 것은 아님

import numpy as np
from tensorflow.keras.datasets import mnist
# from tensorflow.keras.activations import 


# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000,784).astype('float')/255


# 2. 모델
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model


model = autoencoder(hidden_layer_size=104)      # pca 95%

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

model_01 = autoencoder(hidden_layer_size=1)
model_02 = autoencoder(hidden_layer_size=2)
model_04 = autoencoder(hidden_layer_size=4)
model_08 = autoencoder(hidden_layer_size=8)
model_16 = autoencoder(hidden_layer_size=16)
model_32 = autoencoder(hidden_layer_size=32)

print("########################### node 1개 시작 ###########################")
model_01.compile(optimizer='adam', loss='binary_crossentrophy')
model_01.fit(x_train, x_train, epochs=10)

print("########################### node 1개 시작 ###########################")
model_02.compile(optimizer='adam', loss='binary_crossentrophy')
model_02.fit(x_train, x_train, epochs=10)

print("########################### node 1개 시작 ###########################")
model_04.compile(optimizer='adam', loss='binary_crossentrophy')
model_04.fit(x_train, x_train, epochs=10)

print("########################### node 1개 시작 ###########################")
model_08.compile(optimizer='adam', loss='binary_crossentrophy')
model_08.fit(x_train, x_train, epochs=10)

print("########################### node 1개 시작 ###########################")
model_16.compile(optimizer='adam', loss='binary_crossentrophy')
model_16.fit(x_train, x_train, epochs=10)

print("########################### node 1개 시작 ###########################")
model_32.compile(optimizer='adam', loss='binary_crossentrophy')
model_32.fit(x_train, x_train, epochs=10)

output_01 = model_01.predict(x_test)
output_02 = model_02.predict(x_test)
output_04 = model_04.predict(x_test)
output_08 = model_08.predict(x_test)
output_16 = model_16.predict(x_test)
output_32 = model_32.predict(x_test)

from matplotlib import pyplot as plt
import random

fig, axes = plt.subplot(7,5, figsize=(15, 15))

random_images = random.sample(range(output_01.shape[0]), 5)
outputs = [x_test, output_01, output_02, output_04, output_08, output_16, output_32]

# 원본(입력) 이미지를 맨 위에 그린다.
for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_images[col_num]].reshape(28,28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()
