from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from icecream import ic
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


train_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1,
            shear_range=0.5,
            fill_mode='nearest',
)

# xy_train = train_datagen.flow_from_directory(
#             '../_data/brain/train',
#             target_size=(150,150),
#             batch_size=5,
#             class_mode='binary',
#             shuffle=False
# )

'''
1. ImageDataGenerator를 정의
2. flow_from_directory() -> 파일에서 가져오기 - output -> xy_train으로 튜플형태로 뭉쳐 있다.
3. flow() -> 데이터에서 땡겨오기 - output -> x와y가 나뉘어져 있다.

그림 한 개를 가져다가 증폭 (ex. 티셔츠 그림 하나를 반전, 로테이션 등으로 n개로 증폭)



x_train의 0번째 데이터(데이터 불러왔을 때 쉐이프를 맞추기 위해 리쉐잎) 만 100개 만들 것이다(원래 사이즈로 다시 리쉐잎)
y 쉐이프 모양을 맞춰주기 위해 0으로 argument_size(x_train)만큼 맞춰줌
통으로 모두 배치
'''

augment_size = 10

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1),  
    np.zeros(augment_size),
    batch_size=augment_size, 
    shuffle=False
).next() # next() -> iterator친구  # flow, flow_from_drectory => iterator 방식으로 반환 -> 배치사이즈 만큼 리스트로 순서적으로 만들어짐

# ic(type(x_data)) # <class 'tensorflow.python.keras.preprocessing.image.NumpyArrayIterator'> -> .next: <class 'tuple'>
# ic(type(x_data[0])) # <class 'tuple'> -> <class 'numpy.ndarray'>
# ic(type(x_data[0][0])) # <class 'numpy.ndarray'> 
# ic(x_data[0][0].shape) # (100, 28, 28, 1) = x값 
# ic(x_data[0][1].shape) # (100,) = y값
# ic(x_data[0].shape) # (100, 28, 28, 1)
# ic(x_data[1].shape) # (100,)


# 이미지 증폭한 것 확인
import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(10):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')

plt.show()
