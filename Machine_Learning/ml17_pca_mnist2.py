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

pca = PCA(n_components=154)
x = pca.fit_transform(x)

# pca_EVR = pca.explained_variance_ratio_

# cumsum = np.cumsum(pca_EVR)
# ic(cumsum)

# ic(np.argmax(cumsum >= 0.95)+1) # ic| np.argmax(cumsum >= 0.95)+1: 154

# 모델 구성 

# DNN으로 구성하고, 기존 DNN과 비교(PCA한것과 안한 것)

model = Sequential()
model.add(Dense(units=100, activation='relu', input_shape=(28,28)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax')) 

# 전처리


ic(np.unique(y_train)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ic(y_train.shape)            # (60000,1)
ic(y_test.shape) # (10000, 1)
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()


#3. 컴파일, 훈련 metrics=['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(monitor='loss', patience=5, mode='auto', verbose=1)
model.fit(x_train, y_train, epochs=1000, verbose=1, validation_split= 0.001, batch_size=128, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. 평가, 예측 predict 필요x acc로 판단

loss = model.evaluate(x_test, y_test)
ic('loss:', loss[0])
ic('accuracy', loss[1])
ic(f'{걸린시간}분')
# ic(loss)