# categorical_crossentropy와 sigmoid 조합

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
import time



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

test_datagen = ImageDataGenerator(
              rescale=1./255)
            

xy_train = test_datagen.flow_from_directory(
            '../_data/cat_dog/training_set',
            target_size=(150,150),
            batch_size=8100,
            class_mode='categorical',
            )

# Found 8005 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
            '../_data/cat_dog/test_set',
            target_size=(150,150),
            batch_size=2100,
            class_mode='categorical',
            )

# Found 2023 images belonging to 2 classes.



# start = time.time()
# np.save('./_save/_npy/k59_8_cat_dog_train_x.npy', arr=xy_train[0][0])
# np.save('./_save/_npy/k59_8_cat_dog_train_y.npy', arr=xy_train[0][1])
# np.save('./_save/_npy/k59_8_cat_dog_test_x.npy', arr=xy_test[0][0])
# np.save('./_save/_npy/k59_8_cat_dog_test_y.npy', arr=xy_test[0][1])
# save_time = time.time() - start
# ic(save_time)

x_train =np.load('./_save/_npy/k59_8_cat_dog_train_x.npy')
y_train = np.load('./_save/_npy/k59_8_cat_dog_train_y.npy')
x_test = np.load('./_save/_npy/k59_8_cat_dog_test_x.npy')
y_test = np.load('./_save/_npy/k59_8_cat_dog_test_y.npy')

# ic(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

'''
    x_train.shape: (850, 150, 150, 3)
    y_train.shape: (850, 2)
    x_test.shape: (210, 150, 150, 3)
    y_test.shape: (210, 2)

'''




model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(150,150,3), padding='same', activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, steps_per_epoch=32,
                validation_steps=4, validation_split=0.1, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

loss = model.evaluate(x_test, y_test)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
# loss = hist.history['loss']
val_loss = hist.history['val_loss']


ic(걸린시간)

ic(acc[-1])
ic(loss[0])
ic(val_acc[-1])

'''
sigmoid & categorical

ic| 걸린시간: 1.5
ic| acc[-1]: 0.9969461560249329
ic| loss[0]: 2.123854160308838
ic| val_acc[-1]: 0.7203495502471924


'''

