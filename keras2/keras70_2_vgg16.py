from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19


vgg16 = VGG16(weights='imagenet', include_top=True, input_shape=(224,224,3)) 
# include_top=False -> 내가 가진 shape에 맞춰줌 
# include_top=True -> 이미지넷에 맞는 shape로 맞춰야함 
# 이미지 크기를 224,224,3으로 맞춰줘야함

vgg16.trainable=False # vgg 훈련을 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10)) # 나만의 connected layer 만들어줌
model.add(Dense(1))

# model.trainable=False # 전체 모델 훈련을 동결

model.summary()

print(len(model.weights))           # 26 -> 30
print(len(model.trainable_weights)) # 0 -> 4 
# vgg16모델을 훈련안함 풀리커넥티드 (w , b가 하나씩 더 추가 되어서 레이어가2개 늘어나면 4개가 늘어남)


