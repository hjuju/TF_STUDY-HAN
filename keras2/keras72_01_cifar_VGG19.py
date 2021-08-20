from tensorflow.keras import datasets
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar100
from icecream import ic
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import time
from tensorflow.keras.applications import VGG16, VGG19, Xception




(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)
# ic(x_train.shape)
# ic(x_test.shape)



scaler = MinMaxScaler()
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


vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(32,32,3)) 
# include_top=False -> 내가 가진 shape에 맞춰줌 
# include_top=True -> 이미지넷에 맞는 shape로 맞춰야함 
# 이미지 크기를 224,224,3으로 맞춰줘야함

vgg19.trainable=True # vgg 훈련을 동결

model = Sequential()
model.add(vgg19)
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
## Cifar10

# trainable = true, Flatten
loss =  0.8184158205986023
accuracy =  0.7947999835014343

# trainable = False, Flatten
loss =  1.2285866737365723
accuracy =  0.5787000060081482
ic| f'{걸린시간}분': '2.8분'

# trainable = False, GAP
loss =  1.2449361085891724
accuracy =  0.5652999877929688
ic| f'{걸린시간}분': '2.8분'

# trainable = true, GAP
loss =  2.3027846813201904
accuracy =  0.10000000149011612
ic| f'{걸린시간}분': '4.1분'


## Cifar100

# trainable = true, Flatten
loss =  4.6058549880981445
accuracy =  0.009999999776482582
ic| f'{걸린시간}분': '3.6분'

# trainable = False, Flatten
loss =  2.753312587738037
accuracy =  0.328900009393692
ic| f'{걸린시간}분': '2.7분'

# trainable = False, GAP
loss =  2.7605185508728027
accuracy =  0.3278000056743622
ic| f'{걸린시간}분': '1.9분'

# trainable = true, GAP
loss =  4.605761528015137
accuracy =  0.009999999776482582
ic| f'{걸린시간}분': '3.1분'

'''


