import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255

x_test = x_test.reshape(10000,784).astype('float')/255

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense

def autodencoder(hidden_layer_size):
    model = Sequential
    model.add(Dense(unit=hidden_layer_size, input_shape=(784,), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model = autodencoder(hidden_layer_size=154) # pca 95%

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, x_train, epochs=10)



