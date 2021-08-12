import numpy as np
from numpy.core.fromnumeric import reshape
from tensorflow.keras.datasets import mnist
from icecream import ic
from sklearn.decomposition import PCA

(x_train, _), (x_test, _) = mnist.load_data() # _ : 변수로 받지 않는다

ic(x_train.shape, x_test.shape) # x_train.shape: (60000, 28, 28), x_test.shape: (10000, 28, 28)
ic(_.shape)

x = np.append(x_train, x_test, axis=0)
ic(x.shape)# ic| x.shape: (70000, 28, 28)

x = x.reshape(x.shape[0], 28*28)

pca = PCA(n_components=x.shape[1])
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_

cumsum = np.cumsum(pca_EVR)
# ic(cumsum)

ic(np.argmax(cumsum >= 0.95)+1) # ic| np.argmax(cumsum >= 0.95)+1: 154

