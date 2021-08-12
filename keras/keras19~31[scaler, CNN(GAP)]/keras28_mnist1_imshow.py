import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from tensorflow.keras.datasets import mnist
from icecream import ic

(x_train, y_train), (x_test, y_test) = mnist.load_data()

ic(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
ic(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

ic(x_train[0])
ic(y_train[0])

plt.imshow(x_train[0], 'gray')
plt.show()