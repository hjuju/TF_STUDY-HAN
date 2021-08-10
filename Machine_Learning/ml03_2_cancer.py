# 모델 구성하고 완료

import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

datasets = load_breast_cancer()

# ic(datasets.DESCR)
# ic(datasets.feature_names)

x = datasets.data 
y= datasets.target 

ic(x.shape) # (569, 30)
ic(y.shape) # (569,)

ic(y[:10]) 
ic(np.unique(y)) # 데이터가 0과 1로만 이루어져 있으니 이진분류 실시 -> output layer의 활성함수를 sigmoid로 줌

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, shuffle=True, random_state=70)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

from sklearn.svm import LinearSVC, SVC # 먹히는지 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

'''
model = LinearSVC()
ic| result: 0.9692982456140351
ic| r2: 0.8659611992945326

model =SVC()
ic| result: 0.9780701754385965
ic| r2: 0.9042579994960948

model = KNeighborsRegressor()
ic| result: 0.9119173595364072
ic| r2: 0.9119173595364072

model = KNeighborsClassifier()
ic| result: 0.9824561403508771
ic| r2: 0.9234063995968758

model = LogisticRegression()
ic| result: 0.9736842105263158
ic| r2: 0.8851095993953138

model = KNeighborsClassifier()
ic| result: 0.9824561403508771
ic| r2: 0.9234063995968758

model = DecisionTreeClassifier()
ic| result: 0.9429824561403509
ic| r2: 0.7510707986898464

model = DecisionTreeRegressor()
ic| result: 0.7702191987906274
ic| r2: 0.7702191987906274

model = RandomForestRegressor()
ic| result: 0.8562970017636684
ic| r2: 0.8562970017636684

model = RandomForestClassifier()
ic| result: 0.9649122807017544
ic| r2: 0.8468127991937516
'''




#3. 컴파일, 훈련


model.fit(x_train, y_train)


#4. 평가, 예측
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
ic(result)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
# ic(y_predict)
ic(r2)




'''
이진분류
train size = 0.7
model.add(Dense(256, activation='relu', input_shape =(30,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
es = EarlyStopping(monitor='loss', patience=20, mode='auto', verbose=1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='loss', patience=30, mode='auto', verbose=1)
hist = model.fit(x_train, y_train, epochs=1000, verbose=1, batch_size=8, validation_split=0.1, shuffle=True, callbacks=[es])

ic| 'loss:', loss[0]: 0.08795139938592911
ic| 'accuracy', loss[1]: 0.9766082167625427


train size = 0.6
모델 위와 동일
c| 'loss:', loss[0]: 0.1287858933210373
ic| 'accuracy', loss[1]: 0.9868420958518982

'''