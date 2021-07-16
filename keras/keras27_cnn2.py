from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D


model = Sequential()                                         
model.add(Conv2D(10, kernel_size=(2,2),                       # (N, 10, 10, 1)
                    padding='same', input_shape=(10,10,1)))   # (N, 10, 10, 10) 
model.add(Conv2D(20, (2,2), activation='relu'))               # (N, 9, 9, 20(output))
model.add(Conv2D(30, (2,2),  padding='valid'))                # (N, 8, 8, 30)   pooling 후 반으로 줄어듦
model.add(MaxPooling2D())                                     # (N, 4, 4, 30)  (통상적으로 2번 Conv후 pooling 해줌)
model.add(Conv2D(15, (2,2)))                                  # (N, 3, 3, 15)
model.add(Flatten())                                          # (N, 480) 
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 이진분류로 출력


model.summary()