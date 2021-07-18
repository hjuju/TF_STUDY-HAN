import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from icecream import ic
import time

#1. data preprocessing
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train, x_test = x_train / 255, x_test / 255

print(np.unique(y_train)) # [0 1 ... 98 99]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



#2. 모델링
input = Input(shape=(32, 32, 3))
x = Conv2D(64, (3,3), padding='same', activation='relu')(input)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(32, (3,3), activation='relu')(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(100, activation='sigmoid')(x)

model = Model(inputs=input, outputs=output)

#3. 컴파일, 훈련
es = EarlyStopping(monitor='acc', patience=5, mode='auto', verbose=1)
model.compile(loss='binary_crossentropy', optimizer='adam', 
                        metrics=['acc'])
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=128, 
                                validation_split=0.001, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. evaluating, prediction
loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0])
print('accuracy = ', loss[1])
ic(f'{걸린시간}분')