import numpy as np

x_train = np.load('./_save/_npy/k55_x_train_cifar10.npy')
x_test = np.load('./_save/_npy/k55_x_test_cifar10.npy')
y_train = np.load('./_save/_npy/k55_y_train_cifar10.npy')
y_test = np.load('./_save/_npy/k55_y_test_cifar10.npy')

print(x_train.shape) 