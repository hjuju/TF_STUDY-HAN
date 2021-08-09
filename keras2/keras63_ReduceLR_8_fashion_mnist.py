import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. data preprocessing
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))
x_train, x_test = x_train / 255, x_test / 255

# print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. modeling
input = Input(shape=(28, 28, 1))
a = Conv2D(128, (3,3), padding='same', activation='relu')(input)
a = MaxPooling2D((2,2))(a)
a = Conv2D(128, (3,3), padding='same', activation='relu')(a)
a = MaxPooling2D((2,2))(a)
a = Conv2D(64, (3,3), padding='same', activation='relu')(a)
a = Flatten()(a)
a = Dense(128, activation='relu')(a)
a = Dense(64, activation='relu')(a)
a = Dense(32, activation='relu')(a)
output = Dense(10, activation='softmax')(a)

model = Model(inputs=input, outputs=output)



#3. compiling, training
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

optimizer = Adam(lr=0.001) 

es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.01) 

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_split=0.001, callbacks=[es, reduce_lr])

#4. evaluating, prediction
loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0])
print('accuracy = ', loss[1])

'''
reduceLR
loss =  0.28742092847824097
accuracy =  0.9293000102043152

loss =  0.3115113377571106
accuracy =  0.9218999743461609
'''