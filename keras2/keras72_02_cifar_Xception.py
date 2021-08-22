# from tensorflow.keras import datasets
# from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.datasets import cifar100
# from icecream import ic
# from sklearn.preprocessing import OneHotEncoder, MinMaxScaler,RobustScaler
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping
# import time
# from tensorflow.keras.applications import VGG16, VGG19, Xception
# from tensorflow.python.keras.applications import xception
# import numpy as np
# from tensorflow.python.keras.layers.convolutional import UpSampling2D

# size = 32
# (x_train, y_train), (x_test, y_test) = cifar100.load_data()

# x_train = x_train.reshape(50000, size * size * 3)
# x_test = x_test.reshape(10000, size * size * 3)
# # ic(x_train.shape)
# # ic(x_test.shape)

# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# x_train = x_train.reshape(50000, size,  size, 3)
# x_test = x_test.reshape(10000, size,  size,  3)

# one = OneHotEncoder()
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# one.fit(y_train)
# y_train = one.transform(y_train).toarray()
# y_test = one.transform(y_test).toarray()

# ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# tl = Xception(weights='imagenet', include_top=False, input_shape=(96,96,3)) 
# # include_top=False -> 내가 가진 shape에 맞춰줌 
# # include_top=True -> 이미지넷에 맞는 shape로 맞춰야함 
# # 이미지 크기를 224,224,3으로 맞춰줘야함

# tl.trainable=True # vgg 훈련을 동결

# model = Sequential()

# model.add(UpSampling2D(size=(3,3), input_shape=(32,32,3)))
# model.add(tl)
# # model.add(GlobalAveragePooling2D())
# model.add(Flatten())
# model.add(Dense(256)) # 나만의 connected layer 만들어줌
# model.add(Dense(128))
# model.add(Dense(64))
# model.add(Dense(100, activation='softmax'))

# # model.trainable=False # 전체 모델 훈련을 동결

# # model.summary()

# # print(len(model.weights))           # 26 -> 30
# # print(len(model.trainable_weights)) # 0 -> 4 
# # vgg16모델을 훈련안함 풀리커넥티드 (w , b가 하나씩 더 추가 되어서 레이어가2개 늘어나면 4개가 늘어남)

# #3. compiling, training
# es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1)
# model.compile(loss='categorical_crossentropy', optimizer='adam', 
#                         metrics=['acc'])
# start = time.time()
# model.fit(x_train, y_train, epochs=1000, batch_size=128, 
#                                 validation_split=0.1, callbacks=[es], verbose=2)
# 걸린시간 = round((time.time() - start) /60,1)

# #4. evaluating, prediction
# loss = model.evaluate(x_test, y_test)
# y_pred = model.predict(x_test)
# print('loss = ', loss[0])
# print('accuracy = ', loss[1])
# # print(y_pred)
# ic(f'{걸린시간}분')

from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10, cifar100
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG19, Xception, ResNet50, ResNet101, InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import DenseNet121, MobileNetV2, NASNetMobile, EfficientNetB0
from tensorflow.python.keras.layers.convolutional import UpSampling2D
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta, RMSprop, SGD, Nadam
from tensorflow.keras.callbacks import EarlyStopping

COUNT = 1
LOSS_ACC_LS = []
DATASETS = {'cifar_10': cifar10.load_data(), 'cifar100': cifar100.load_data()}
TRAINABLE = {'True_': True, 'False': False}
FLATTEN_GAP = {'Flatten': Flatten(), 'GAP__2D': GlobalAveragePooling2D()}

for dt_key, dt_val in DATASETS.items():
    #1 Data
    (x_train, y_train), (x_test, y_test) = dt_val
    x_train = x_train.reshape(-1, 32 * 32 * 3)
    x_test = x_test.reshape(-1, 32 * 32 * 3)
    # ic(x_train.shape, x_test.shape)
    # ic(np.unique(y_train))

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = x_train.reshape(-1, 32, 32, 3)
    x_test = x_test.reshape(-1, 32, 32, 3)
    
    #2 Model
    for tf_key, tf_val in TRAINABLE.items():
        for fg_key, fg_val in FLATTEN_GAP.items():
            transfer_learning = Xception(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
            transfer_learning.trainable = tf_val

            model = Sequential()
            model.add(UpSampling2D(size=(3,3), input_shape=(32,32,3)))
            model.add(transfer_learning)
            model.add(fg_val)
            if dt_key == 'cifar10':
                model.add(Dense(100, activation='relu'))
                model.add(Dense(50, activation='relu'))
                model.add(Dense(10, activation='softmax'))
            else:
                model.add(Dense(200, activation='relu'))
                model.add(Dense(100, activation='softmax'))
            # model.summary()

            #3 Train
            opt = Adam()
            model.compile(loss='sparse_categorical_crossentropy',
                        optimizer=opt, metrics=['acc'])
            es = EarlyStopping(monitor='val_loss', patience=4, mode='min', verbose=1)
            model.fit(x_train, y_train, epochs=20, batch_size=64,
                    verbose=2, validation_split=0.25, callbacks=[es])

            #4 Evaluate
            loss = model.evaluate(x_test, y_test, batch_size=128)
            result = f'[{COUNT}] {dt_key}_{tf_key}_{fg_key} :: loss= {round(loss[0], 4)}, acc= {round(loss[1], 4)}'
            ic(result)
            LOSS_ACC_LS.append(result)
            COUNT = COUNT + 1

print('Xception')

for i in LOSS_ACC_LS:
    print(i)

'''
[1] cifar_10_True__Flatten :: loss= 0.3428, acc= 0.9089
[2] cifar_10_True__GAP__2D :: loss= 0.5011, acc= 0.8644
[3] cifar_10_False_Flatten :: loss= 1.0845, acc= 0.6967
[4] cifar_10_False_GAP__2D :: loss= 0.9144, acc= 0.7143
[5] cifar100_True__Flatten :: loss= 1.9819, acc= 0.5952
[6] cifar100_True__GAP__2D :: loss= 1.8518, acc= 0.6282
[7] cifar100_False_Flatten :: loss= 2.511, acc= 0.4025
[8] cifar100_False_GAP__2D :: loss= 2.3905, acc= 0.4541

'''


