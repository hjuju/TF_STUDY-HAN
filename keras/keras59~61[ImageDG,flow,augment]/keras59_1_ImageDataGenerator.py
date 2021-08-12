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
 rotation_range=40,      # 랜덤하게 사진을 회전시킬 각도 범위(1-180도)
  width_shift_range=0.2,  # 사진을 수평으로 랜덤하게 평행이동시킬 범위
  height_shift_range=0.2, # 사진을 수직으로 랜덤하게 평행이동시킬 범위
  shear_range=0.2,        # 랜덤하게 전단변환을 적용할 각도 범위
  zoom_range=0.2,         # 랜덤하게 사진을 확대할 범위
  horizontal_flip=True,   # 랜덤하게 이미지를 수평으로 뒤집을지 여부
  fill_mode='nearest',    # 회전/이동 등으로 새롭게 생성되는 픽셀을 채울 방법

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
# (5, 150, 150, 3) (5,) 5 -> 배치사이즈

# ic(xy_train[31][1]) # -> 5장씩(배치사이즈 만큼) 31번까지(총 32개)(이미지개수/배치사이즈) 만큼 생김

