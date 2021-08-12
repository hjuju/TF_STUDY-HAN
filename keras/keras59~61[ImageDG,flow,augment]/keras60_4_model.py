from numpy.lib.type_check import imag
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder
from icecream import ic
from tensorflow.python.keras.layers import Dropout, GlobalAveragePooling2D


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


train_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1,
            shear_range=0.5,
            fill_mode='nearest',
)

# xy_train = train_datagen.flow_from_directory(
#             '../_data/brain/train',
#             target_size=(150,150),
#             batch_size=5,
#             class_mode='binary',
#             shuffle=False
# )


augment_size = 40000

randidx = np.random.randint(x_train.shape[0], size=augment_size)
# ic(x_train.shape[0]) # 60000
# ic(randidx) # [25219, 55054, 31576, ...,  4870, 47427, 44632]
# ic(randidx.shape) # (40000,)

x_augumented= x_train[randidx].copy() # 이미지제너레이터의 flow로 불러와 변경해줌
y_augumented= y_train[randidx].copy() # y값은 라벨값이기 때문에
# ic(x_augumented.shape) # (40000, 28, 28)


x_augumented = x_augumented.reshape(x_augumented.shape[0], 28, 28, 1) # flow는 4차원으로 받아줘야함
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# ic(x_augumented)
 
x_augumented = train_datagen.flow(x_augumented, np.zeros(augment_size),
                                     batch_size=augment_size, shuffle=False).next()[0] # .next()[0] x만 출력

# ic(x_augumented.shape) # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))

# ic(x_train.shape, y_train.shape) # (100000, 28, 28, 1), (100000,)

# one = OneHotEncoder()
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# ic(y_train.shape)            # (60000,1)
# ic(y_test.shape) # (10000, 1)
# one.fit(y_train)
# y_train = one.transform(y_train).toarray()
# y_test = one.transform(y_test).toarray()


# 4만장의 데이터가 순서가 변경되지 않은채 그대로 받음 np.zeros 자리에 y_augmented를 넣어줘도 됨

# loss, val_loss, acc, val_acc 4만개 증폭한것에 대해 과적합과 acc가 좋아졌는지 확인 / 기존 fashion_mnist 결과비교

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
import time

model = Sequential()
model.add(Conv2D(126, kernel_size=(2,2), padding='same', input_shape=(28, 28, 1)))
model.add(MaxPool2D(2,2))
model.add(Conv2D(256, (2,2), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(512, (2,2), activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

es = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
hist = model.fit(x_train, y_train, batch_size=128, epochs=1000, validation_split=0.1, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

loss = model.evaluate(x_test, y_test)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
# loss = hist.history['loss']
val_loss = hist.history['val_loss']


ic(걸린시간)

ic(acc[-1])
ic(loss[0])
ic(loss[1])
ic(val_acc[-1])


'''

증폭X 
ic| 'loss:', loss[0]: 0.02277880162000656
ic| 'accuracy', loss[1]: 0.9912999868392944

증폭O, sparse_categorical_crossentropy
ic| 걸린시간: 3.7
ic| acc[-1]: 0.9739221930503845
ic| loss[0]: 0.761974573135376
ic| val_acc[-1]: 0.7853000164031982

증폭O, categorical_crossentropy + 원핫인코더 
ic| 걸린시간: 6.4
ic| acc[-1]: 0.8854444622993469
ic| loss[0]: 0.5009903907775879
ic| val_acc[-1]: 0.7702999711036682


'''