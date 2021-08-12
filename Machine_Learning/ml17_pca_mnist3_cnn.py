import numpy as np
from numpy.core.fromnumeric import reshape
from tensorflow.keras.datasets import mnist
from icecream import ic
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling3D
from sklearn.model_selection import train_test_split
import time

from tensorflow.python.keras.layers.pooling import MaxPooling2D

(x_train, y_train), (x_test, y_test) = mnist.load_data() # _ : 변수로 받지 않는다

ic(x_train.shape, x_test.shape) # x_train.shape: (60000, 28, 28), x_test.shape: (10000, 28, 28)


x = np.append(x_train, x_test, axis=0)
ic(x.shape)# ic| x.shape: (70000, 28, 28)
y = np.append(y_train, y_test, axis=0)


compoents = 28*28

x = x.reshape(x.shape[0], 28*28)
# x.shape[1]
pca = PCA(n_components=compoents)
x = pca.fit_transform(x)

# pca_EVR = pca.explained_variance_ratio_

# cumsum = np.cumsum(pca_EVR)
# ic(cumsum)

# ic(np.argmax(cumsum >= 0.999)+1) # ic| np.argmax(cumsum >= 0.999)+1: 486


x = x.reshape(x.shape[0], 28,28,1)


ic(x.shape) # ic| x.shape: (70000, 486)

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, train_size=0.8, random_state=66)

ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape, np.unique(y_train))

'''
ic| x_train.shape: (56000, 486, 1, 1)
    x_test.shape: (14000, 486, 1, 1)

'''


# 모델 구성 

# DNN으로 구성하고, 기존 DNN과 비교(PCA한것과 안한 것)

model = Sequential()
model.add(Conv2D(64, kernel_size=(2,2), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (2,2), padding='same',activation='relu')) 
# model.add(MaxPooling2D(2,padding='same'))            
model.add(Conv2D(32, (2,2),padding='same', activation='relu'))
model.add(Conv2D(16, (2,2),padding='same', activation='relu'))                   
model.add(Conv2D(16, (2,2),padding='same', activation='relu'))                                                        
model.add(Flatten())                                          
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax')) 

# 전처리


# ic(np.unique(y_train)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


#3. 컴파일, 훈련 metrics=['acc']
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(monitor='loss', patience=5, mode='auto', verbose=1)
model.fit(x_train, y_train, epochs=1000, verbose=1, validation_split= 0.1, batch_size=128, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. 평가, 예측 predict 필요x acc로 판단

loss = model.evaluate(x_test, y_test)
ic('loss:', loss[0])
ic('accuracy', loss[1])
ic(f'{걸린시간}분')
# ic(loss)

'''
# PCA(486)
# ic| 'loss:', loss[0]: 0.2959951162338257
# ic| 'accuracy', loss[1]: 0.9559285640716553
# ic| f'{걸린시간}분': '0.2분'

PCA(X)
ic| 'loss:', loss[0]: 0.35307228565216064
ic| 'accuracy', loss[1]: 0.9562143087387085
ic| f'{걸린시간}분': '4.5분'

'''