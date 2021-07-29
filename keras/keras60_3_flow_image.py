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

x_augmented= x_train[randidx].copy() # 이미지제너레이터의 flow로 불러와 변경해줌
y_augmented= y_train[randidx].copy() # y값은 라벨값이기 때문에
# ic(x_augumented.shape) # (40000, 28, 28)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 28, 28, 1) # flow는 4차원으로 받아줘야함
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# ic(x_augumented)
 
x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                     batch_size=augment_size, shuffle=False).next()[0] # .next()[0] x만 출력

# ic(x_augumented.shape) # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

ic(x_train.shape, y_train.shape) # (100000, 28, 28, 1), (100000,)



# 4만장의 데이터가 순서가 변경되지 않은채 그대로 받음 np.zeros 자리에 y_augmented를 넣어줘도 됨

# 실습 1. x_augumented 10개가 원래 X_train 10개를 비교하는 이미지를 출력할 것
# subplot(2,10,?)사용

# x_train [25219, 55054, 31576, ...,  4870, 47427, 44632] 앞에서 10개
# x_augumented [25219, 55054, 31576, ...,  4870, 47427, 44632] 앞에서 10개

augment_size = 50


# x_data = train_datagen.flow(
#     np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1),  
#     np.zeros(augment_size),
#     batch_size=augment_size, 
#     shuffle=False)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 2))
for i in range(20):
    plt.subplot(2, 10, i+1)
    plt.axis('off')
    if i < 10:
        plt.imshow(x_train[i], cmap='gray')
    else:
        plt.imshow(x_augmented[i-10], cmap='gray')
plt.show()