# 훈련데이터를 기존데이터 + 20% 더
# 성과비교
# save_dir도 temp에 넣을 것
# 증폭데이터는 temp에 저장 후 훈련끝나고 삭제


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from icecream import ic

# 1. data

x_train = np.load('./_save/_npy/k59_5_gender_train_x.npy')
# y_train = np.load('./_save/_npy/k59_5_gender_train_y.npy')
x_test = np.load('./_save/_npy/k59_5_gender_test_x.npy')
# y_test = np.load('./_save/_npy/k59_5_gender_test_y.npy')

ic(x_train.shape, x_test.shape) # x_train.shape: (3048, 150, 150, 3), x_test.shape: (661, 150, 150, 3)
# y_train = np.concatenate((y_train, y_argmented)) # (100000,)

x_train = x_train.reshape(2648, 67500).astype('float')/255
x_test = x_test.reshape(661, 67500).astype('float')/255

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape) # x_train(0~1) 에 랜덤하게 정규분표로부터 0~0.1의 값을 넣음 
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) # 최소값을 벗어나는 것은 0, 최대값은 1로 한다
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)


# x_train = x_train.reshape(x_train.shape[0], 150, 150, 3)
# x_test = x_test.reshape(x_test.shape[0], 150, 150, 3)
x_train_noised = x_train_noised.reshape(x_train_noised.shape[0], 150, 150, 3)
x_test_noised = x_test_noised.reshape(x_test_noised.shape[0], 150, 150, 3)

ic(x_train_noised.shape, x_test_noised.shape, x_train.shape, x_test.shape)


# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAvgPool2D


def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(GlobalAvgPool2D())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    return model


model = autoencoder(hidden_layer_size=154)      # pca 95%

model.compile(optimizer='adam', loss='mse')

model.fit(x_train_noised, x_train, epochs=10) # 노이즈가 있는것과 없는것을 훈련시킴

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax11, ax12, ax13, ax14, ax15), (ax6, ax7, ax8, ax9, ax10)) = \
        plt.subplots(3, 5, figsize=(15, 7))


# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(150, 150, 3), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

# 노이를 넣은 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noised[random_images[i]].reshape(150, 150, 3), cmap='gray')
    if i == 0:
        ax.set_ylabel("NOISE", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(150, 150, 3), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()