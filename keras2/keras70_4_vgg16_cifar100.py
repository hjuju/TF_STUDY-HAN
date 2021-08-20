from tensorflow.keras import datasets
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.datasets import cifar100
from icecream import ic
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
import time



(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)
# ic(x_train.shape)
# ic(x_test.shape)



scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32,  32, 3)
x_test = x_test.reshape(10000, 32,  32,  3)

one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()

ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3)) 
# include_top=False -> 내가 가진 shape에 맞춰줌 
# include_top=True -> 이미지넷에 맞는 shape로 맞춰야함 
# 이미지 크기를 224,224,3으로 맞춰줘야함

vgg16.trainable=True # False일때 vgg 훈련을 동결

model = Sequential()
model.add(vgg16)
# model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(256)) # 나만의 connected layer 만들어줌
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(100, activation='softmax'))

# model.trainable=False # 전체 모델 훈련을 동결

# model.summary()

print(len(model.weights))           # 26 -> 30
print(len(model.trainable_weights)) # 0 -> 4 
# vgg16모델을 훈련안함 풀리커넥티드 (w , b가 하나씩 더 추가 되어서 레이어가2개 늘어나면 4개가 늘어남)

#3. compiling, training
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                        metrics=['acc'])
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=128, 
                                validation_split=0.1, callbacks=[es])
걸린시간 = round((time.time() - start) /60,1)

#4. evaluating, prediction
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
print('loss = ', loss[0])
print('accuracy = ', loss[1])
# print(y_pred)
ic(f'{걸린시간}분')

'''
vgg16.trainable=False, GAP

loss =  2.4882121086120605
accuracy =  0.3797000050544739
ic| f'{걸린시간}분': '1.6분'

vgg16.trainable=True, GAP

loss =  3.0443339347839355
accuracy =  0.3783000111579895
ic| f'{걸린시간}분': '5.7분'

vgg16.trainable=False, Flatten
loss =  2.4655117988586426
accuracy =  0.38339999318122864
ic| f'{걸린시간}분': '1.4분'

vgg16.trainable=True, Flatteb
loss =  0.8746398687362671
accuracy =  0.8040000200271606

'''


