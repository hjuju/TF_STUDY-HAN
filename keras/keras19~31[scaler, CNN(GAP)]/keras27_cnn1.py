from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten  


model = Sequential()                                          # (N, 5, 5, 1)
model.add(Conv2D(10, kernel_size=(2,2), padding='same', input_shape=(5,5,1))) # (N, 4, 4, 10) 행무시 -> 전체 데이터의 크기 -> 여기선 batch_size
# padding='same'을 주면 이미지 규격 자체로 전달(N, 5, 5, 1)
# (batch_size, height, width) // kernel_size= 2*2로 자를 것 이다 / (28,28,1) 28*28사이즈의 흑백 이미지 => (4,4,10(장))으로 변환
# kernel_size= 를 명시 안해도 가능 (3,3,20)으로 변환 (중요한 특성은 강하게 남고, 약한 특성은 소멸) 수치화 해서 특정 부분의 특성을 저장
model.add(Conv2D(20, (2,2), activation='relu')) # (N, 3, 3, 20(노드의 개수))
model.add(Conv2D(30, (2,2), activation='relu'))
model.add(Flatten()) # (N, 180) 다차원 데이터를 2차원 데이터로 변환 플래튼 다음에는 Dense로 신경망 만듦
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 이진분류로 출력


model.summary()