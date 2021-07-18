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
x1 = Conv2D(128, (3,3), padding='same', activation='relu')(input)
x2 = MaxPooling2D((2,2))(x1)
x3 = Conv2D(128, (3,3), padding='same', activation='relu')(x2)
x4 = MaxPooling2D((2,2))(x3)
x5 = Conv2D(64, (3,3), padding='same', activation='relu')(x4)
x6 = Flatten()(x5)
x7 = Dense(64, activation='relu')(x6)
x7 = Dense(32, activation='relu')(x6)
output = Dense(10, activation='softmax')(x7)

model = Model(inputs=input, outputs=output)

#3. compiling, training
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                        metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=128, 
                                validation_split=0.001, callbacks=[es])

#4. evaluating, prediction
loss = model.evaluate(x_test, y_test)

print('loss = ', loss[0])
print('accuracy = ', loss[1])

'''
loss =  0.3116116523742676
accuracy =  0.9196000099182129
'''