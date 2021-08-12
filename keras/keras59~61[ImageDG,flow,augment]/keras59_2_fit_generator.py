import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from icecream import ic


train_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=5,
            zoom_range=1.2,
            shear_range=0.7,
            fill_mode='nearest',
)

test_datagen = ImageDataGenerator(rescale=1./255)

'''
수치화, 데이터증폭 rescale- 0~255사이로 되어있으니 0~1사이로 수치화
horizontal_flip 수평이동
vertical_flip 수직이동
이미지 증폭 시 좌우 상하 이동
width_shift_range
height_shift_range
rotation_range
zoom_range=1.2 (1.2배 더 크게)
shear_range =?
fill_mode 공백을 가까운이미지로 채워넣음

train은 증폭을 시킬 수 있지만 test데이터는 증폭하지 않음

'''

xy_train = train_datagen.flow_from_directory(
            '../_data/brain/train',
            target_size=(150,150),
            batch_size=5,
            class_mode='binary',
            shuffle=False
)
# IDG만 실행하면 나오는 문구: Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
            '../_data/brain/test',
            target_size=(150,150),
            batch_size=5,
            class_mode='binary',
)
# Found 120 images belonging to 2 classes.

'''
이 폴더에 있는 이미지 사이즈를 일괄적으로 똑같은 사이즈로 조정
class_mode 라벨값이 정상, 비정상으로 (0,1)로 라벨링 / 폴더에는 동일한 라벨값으로 나눠서 함께 넣어줌 / 여러개는 categorical
batch_size -> y값 나오는 개수

'''

# ic(xy_train) # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001876E888550>
# ic(type(xy_train[0])) # y값으로 나오는 것은 배치사이즈 [0., 1., 0., 1., 1.] ic(xy_train[0][1])
# ic(type(xy_train[0][0])) # x값 type(xy_train[0][0]): <class 'numpy.ndarray'>
# ic(type(xy_train[0][1])) # y값 type(xy_train[0][1]): <class 'numpy.ndarray'>
# ic(xy_train[0][2]) # 없음
# ic(xy_train[0][0].shape, xy_train[0][1].shape) 
# (5, 150, 150, 3) (5,) 5 -> 배치사이즈 / 디폴트를 컬러로 인식

# ic(xy_train[31][1]) # -> 5장씩(배치사이즈 만큼) 31번까지(총 32개)(이미지개수/배치사이즈) 만큼 생김

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(128, (2,2), input_shape=(150,150,3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# hist = model.fit_generator(xy_train, epoch=50, steps_per_epoch=32, validation_data=xy_test, validation_steps=4) 
# # x와y가 튜플로 묶여있는 데이터의 훈련 , steps_per_epoch-> 1에포에 배치사이즈 몇?(이미지개수/배치사이즈) / validation_data도 쌍으로 묶인 데이터 넣어줌
# # validation_steps
# acc = hist.history['acc']
# val_acc = hist.history['val_acc']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# # 시각화

# ic(acc[-1])
# ic(val_acc[:-1])
# ic()

