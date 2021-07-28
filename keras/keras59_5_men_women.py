# 1. 남여 데이터로 모델링 구성

# 2 본인사진으로 프레딕트

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
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

pred_datagen = ImageDataGenerator(
              rescale=1./255,   
              horizontal_flip=True,
              vertical_flip=True,
              width_shift_range=0.1,
              height_shift_range=0.1,
              rotation_range=5,
              zoom_range=1.2,
              shear_range=0.7,
              fill_mode='nearest')
            

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

x_prd = pred_datagen.flow_from_directory('../_data/men_women/men_women_prd/',
              target_size=(150,150),
              batch_size=100
              
             
)

# Found 661 images belonging to 2 classes

# ic(xy_train[0][1])
# ic(xy_test[0][1])


# np.save('./_save/_npy/k59_5_gender_train_x.npy', arr=xy_train[0][0])
# np.save('./_save/_npy/k59_5_gender_train_y.npy', arr=xy_train[0][1])
# np.save('./_save/_npy/k59_5_gender_test_x.npy', arr=xy_test[0][0])
# np.save('./_save/_npy/k59_5_gender_test_y.npy', arr=xy_test[0][1])
# np.save('./_save/_npy/k59_5_gender_x_prd.npy', arr=x_prd[0][0])

# ic(x_prd[0][0])

x_train = np.load('./_save/_npy/k59_5_gender_train_x.npy')
y_train = np.load('./_save/_npy/k59_5_gender_train_y.npy')
x_test = np.load('./_save/_npy/k59_5_gender_test_x.npy')
y_test = np.load('./_save/_npy/k59_5_gender_test_y.npy')



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
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, steps_per_epoch=32,
                validation_steps=4, validation_split=0.1, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

# img = image.load_img('../_data/men_women/prd/001.jpg', target_size=(150,150))
# prd_img = image.img_to_array(img)
# prd_img = np.expand_dims(prd_img, axis=0)

loss = model.evaluate(x_test, y_test)
result = model.predict([x_prd])

acc = hist.history['acc']
val_acc = hist.history['val_acc']
# loss = hist.history['loss']
val_loss = hist.history['val_loss']

result = (1- result) * 100

ic(걸린시간)

ic(acc[-1])
ic(loss[0])
ic(val_acc[-1])
ic(result)

'''
# ic| 걸린시간: 0.6
# ic| acc[-1]: 0.989928662776947
# ic| loss[0]: 4.575331687927246
# ic| val_acc[-1]: 0.6188679337501526
'''


