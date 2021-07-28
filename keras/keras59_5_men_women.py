# 1. 남여 데이터로 모델링 구성

# 2 본인사진으로 프레딕트

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from icecream import ic

# img_datagen = ImageDataGenerator(
#             rescale=1./255,
#             horizontal_flip=True,
#             vertical_flip=True,
#             width_shift_range=0.1,
#             height_shift_range=0.1,
#             rotation_range=5,
#             zoom_range=1.2,
#             shear_range=0.7,
#             fill_mode='nearest',
#             validation_split=0.2)

pred_datagen = ImageDataGenerator(rescale=1./255,
              rescale=1./255,
              horizontal_flip=True,
              vertical_flip=True,
              width_shift_range=0.1,
              height_shift_range=0.1,
              rotation_range=5,
              zoom_range=1.2,
              shear_range=0.7,
              fill_mode='nearest',)
            

# xy_train = img_datagen.flow_from_directory(
#             '../_data/men_women',
#             target_size=(150,150),
#             batch_size=3000,
#             class_mode='binary',
#             subset='training')

# # Found 2648 images belonging to 2 classes.

# xy_test = img_datagen.flow_from_directory(
#             '../_data/men_women',
#             target_size=(150,150),
#             batch_size=700,
#             class_mode='binary',
#             subset='validation')

x_prd = pred_datagen.flow_from_directory(
              '../_data/men_women/my',
              target_size=(150,150),
              batch_size=10,
              class_mode='binary'
)

# Found 661 images belonging to 2 classes

# ic(xy_train[0][1])
# ic(xy_test[0][1])


# np.save('./_save/_npy/k59_5_gender_train_x.npy', arr=xy_train[0][0])
# np.save('./_save/_npy/k59_5_gender_train_y.npy', arr=xy_train[0][1])
# np.save('./_save/_npy/k59_5_gender_test_x.npy', arr=xy_test[0][0])
# np.save('./_save/_npy/k59_5_gender_test_y.npy', arr=xy_test[0][1])

# x_train = np.load('./_save/_npy/k59_5_gender_train_x.npy')
# y_train = np.load('./_save/_npy/k59_5_gender_train_y.npy')
# x_test = np.load('./_save/_npy/k59_5_gender_test_x.npy')
# y_test = np.load('./_save/_npy/k59_5_gender_test_y.npy')



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(128, (3,3), input_shape=(150,150,3), padding='same'))
model.add(Conv2D(128, (3,3), input_shape=(150,150,3), padding='same'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()



