import numpy as np

x_train = np.load('./_save/_npy/k55_x_train_fashion_mnist.npy')
x_test = np.load('./_save/_npy/k55_x_test_fashion_mnist.npy')
y_train = np.load('./_save/_npy/k55_y_train_fashion_mnist.npy')
y_test = np.load('./_save/_npy/k55_y_test_fashion_mnist.npy')

print(x_train.shape) 