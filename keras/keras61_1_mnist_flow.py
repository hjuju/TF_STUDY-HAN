# 훈련데이터를 총 10만개로 증폭!
# 완료 후 기존 모델과 비교
# save_dir도 temp에 넣을 것
# 증폭데이터는 temp에 저장 후 훈련끝나고 삭제

from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from icecream import ic

(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1,
            shear_range=0.5,
            fill_mode='nearest')

# ic(x_train.shape) # (60000, 28, 28)

augment_size = 10

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1),  
    np.zeros(augment_size),
    batch_size=augment_size, 
    shuffle=False).next()

