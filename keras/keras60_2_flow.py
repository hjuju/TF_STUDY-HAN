from numpy.lib.type_check import imag
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from icecream import ic
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

ic(x_train.shape, y_train.shape) # (100000, 28, 28, 1), (100000,)



# 4만장의 데이터가 순서가 변경되지 않은채 그대로 받음 np.zeros 자리에 y_augmented를 넣어줘도 됨

