import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

# pred_datagen = ImageDataGenerator(rescale=1./255,
#               rescale=1./255,
#               horizontal_flip=True,
#               vertical_flip=True,
#               width_shift_range=0.1,
#               height_shift_range=0.1,
#               rotation_range=5,
#               zoom_range=1.2,
#               shear_range=0.7,
#               fill_mode='nearest',)
            

# xy_train = img_datagen.flow_from_directory(
#             '../_data/rps/',
#             target_size=(150,150),
#             batch_size=20,
#             class_mode='categorical',
#             subset='training')

# # Found 2016 images belonging to 3 classes.

# xy_test = img_datagen.flow_from_directory(
#             '../_data/rps/',
#             target_size=(150,150),
#             batch_size=20,
#             class_mode='categorical',
#             subset='validation')

# Found 504 images belonging to 3 classes.

# np.save('./_save/_npy/k59_6_rps_train_x.npy', arr=xy_train[0][0])
# np.save('./_save/_npy/k59_6_rps_train_y.npy', arr=xy_train[0][1])
# np.save('./_save/_npy/k59_6_rps_test_x.npy', arr=xy_test[0][0])
# np.save('./_save/_npy/k59_6_rps_test_y.npy', arr=xy_test[0][1])

x_train =np.load('./_save/_npy/k59_6_rps_train_x.npy')
y_train = np.load('./_save/_npy/k59_6_rps_train_y.npy')
x_test = np.load('./_save/_npy/k59_6_rps_test_x.npy')
y_test = np.load('./_save/_npy/k59_6_rps_test_y.npy')

# ic(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
import time

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(150,150,3), padding='same', activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))

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
ic| 걸린시간: 0.5
ic| acc[-1]: 0.9614112377166748
ic| loss[0]: 3.1957826614379883
ic| val_acc[-1]: 0.5346534848213196
'''
