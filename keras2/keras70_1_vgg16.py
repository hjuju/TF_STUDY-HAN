from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19

model = VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3)) 
# include_top=False -> 내가 가진 shape에 맞춰줌 
# include_top=True -> 이미지넷에 맞는 shape로 맞춰야함 
# 이미지 크기를 224,224,3으로 맞춰줘야함

model.trainable=True 
# weight값을 그대로 쓰겠다(이미지넷 레이어로 훈련하지 않겠다. weight의 갱신이없다) -> model.predict에서 사용 / 시간이 빠름

model.summary()

print(len(model.weights))
print(len(model.trainable_weights))


'''
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
_________________________________________________________________
.....................................
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000
=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0


# fc

1. Convolution/Pooling 메커니즘은 이미지를 형상으로 분할하고 분석.

2. FC(Fully Connected Layer)로, 이미지를 분류/설명하는 데 가장 적합하게 예측


완전히 연결 되었다라는 뜻으로, 한층의 모든 뉴런이 다음층이 모든 뉴런과 연결된 상태로

2차원의 배열 형태 이미지를 1차원의 평탄화 작업을 통해 이미지를 분류하는데 사용되는 계층

1. 2차원 배열 형태의 이미지를 1차원 배열로 평탄화

2. 활성화 함수(Relu, Leaky Relu, Tanh 등)뉴런을 활성화

3. 분류기(Softmax) 함수로 분류




'''