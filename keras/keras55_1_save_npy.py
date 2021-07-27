from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_diabetes, load_wine
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np
from tensorflow.keras import datasets
from icecream import ic

datasets = load_iris()

x_data_iris = datasets.data
y_data_iris = datasets.target

np.save("./_save/_npy/k55_x_data_iris.npy", arr=x_data_iris) # 넘파이로 정해준 경로에 데이터 세이브
np.save("./_save/_npy/k55_y_data_iris.npy", arr=y_data_iris)


############################################ boston ############################################


datasets = load_boston()

x_data_boston = datasets.data
y_data_boston = datasets.target

np.save("./_save/_npy/k55_x_data_boston.npy", arr=x_data_boston) 
np.save("./_save/_npy/k55_y_data_boston.npy", arr=y_data_boston)


############################################ cancer ############################################

datasets = load_breast_cancer()

x_data_cancer = datasets.data
y_data_cancer = datasets.target

np.save("./_save/_npy/k55_x_data_cancer.npy", arr=x_data_cancer) 
np.save("./_save/_npy/k55_y_data_cancer.npy", arr=y_data_cancer)


############################################ diabetes ############################################


datasets = load_diabetes()

x_data_diabetes = datasets.data
y_data_diabetes = datasets.target

np.save("./_save/_npy/k55_x_data_diabetes.npy", arr=x_data_diabetes) 
np.save("./_save/_npy/k55_y_data_diabetes.npy", arr=y_data_diabetes)


############################################ wine ############################################


datasets = load_wine()

x_data_wine = datasets.data
y_data_wine = datasets.target

np.save("./_save/_npy/k55_x_data_wine.npy", arr=x_data_wine) 
np.save("./_save/_npy/k55_y_data_wine.npy", arr=y_data_wine)

############################################ mnist ############################################


(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train_minist = x_train
x_test_minist = x_test
y_train_minist = y_train
y_test_minist = y_test

np.save("./_save/_npy/k55_x_train_mnist", arr=x_train_minist)
np.save("./_save/_npy/k55_x_test_mnist", arr=x_test_minist) 
np.save("./_save/_npy/k55_y_train_mnist", arr=y_train_minist)
np.save("./_save/_npy/k55_y_test_mnist", arr=y_test_minist)

############################################ f_mnist ############################################


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


x_train_fashion_minist = x_train
x_test_fashion_minist = x_test
y_train_fashion_minist = y_train
y_test_fashion_minist = y_test

np.save("./_save/_npy/k55_x_train_fashion_mnist", arr=x_train_fashion_minist)
np.save("./_save/_npy/k55_x_test_fashion_mnist", arr=x_test_fashion_minist) 
np.save("./_save/_npy/k55_y_train_fashion_mnist", arr=y_train_fashion_minist)
np.save("./_save/_npy/k55_y_test_fashion_mnist", arr=y_test_fashion_minist)

############################################ cifar10 ############################################


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


x_train_cifar10 = x_train
x_test_cifar10 = x_test
y_train_cifar10 = y_train
y_test_cifar10 = y_test

np.save("./_save/_npy/k55_x_train_cifar10", arr=x_train_cifar10)
np.save("./_save/_npy/k55_x_test_cifar10", arr=x_test_cifar10) 
np.save("./_save/_npy/k55_y_train_cifar10", arr=y_train_cifar10)
np.save("./_save/_npy/k55_y_test_cifar10", arr=y_test_cifar10)

############################################ cifar100 ############################################


(x_train, y_train), (x_test, y_test) = cifar100.load_data()


x_train_cifar100 = x_train
x_test_cifar100 = x_test
y_train_cifar100 = y_train
y_test_cifar100 = y_test

np.save("./_save/_npy/k55_x_train_cifar100", arr=x_train_cifar100)
np.save("./_save/_npy/k55_x_test_cifar100", arr=x_test_cifar100) 
np.save("./_save/_npy/k55_y_train_cifar100", arr=y_train_cifar100)
np.save("./_save/_npy/k55_y_test_cifar100", arr=y_test_cifar100)



