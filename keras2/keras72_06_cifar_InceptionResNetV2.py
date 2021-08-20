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
            transfer_learning = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
            transfer_learning.trainable = tf_val

            model = Sequential()
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
            opt = Adam(0.0001)
            model.compile(loss='sparse_categorical_crossentropy',
                        optimizer=opt, metrics=['acc'])
            es = EarlyStopping(monitor='val_loss', patience=4, mode='min', verbose=1)
            model.fit(x_train, y_train, epochs=20, batch_size=512,
                    verbose=1, validation_split=0.25, callbacks=[es])

            #4 Evaluate
            loss = model.evaluate(x_test, y_test, batch_size=128)
            result = f'[{COUNT}] {dt_key}_{tf_key}_{fg_key} :: loss= {round(loss[0], 4)}, acc= {round(loss[1], 4)}'
            ic(result)
            LOSS_ACC_LS.append(result)
            COUNT = COUNT + 1

print('InceptionResNetV2')

for i in LOSS_ACC_LS:
    print(i)

'''
[1] cifar_10_True__Flatten :: loss= 5.3678, acc= 0.1642
[2] cifar_10_True__GAP__2D :: loss= 3.7765, acc= 0.3044
[3] cifar_10_False_Flatten :: loss= 1.807, acc= 0.3567
[4] cifar_10_False_GAP__2D :: loss= 1.8083, acc= 0.3577
[5] cifar100_True__Flatten :: loss= 5.0668, acc= 0.1776
[6] cifar100_True__GAP__2D :: loss= 5.0931, acc= 0.1691
[7] cifar100_False_Flatten :: loss= 4.0988, acc= 0.0955
[8] cifar100_False_GAP__2D :: loss= 4.0982, acc= 0.0958

'''