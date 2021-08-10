import numpy as np
from sklearn.datasets import load_iris
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore') # 경고무시

### 머신러닝(evaluate -> score)

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

# 1. 데이터
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)  # (150, 4), (150,)->(150, 3)
ic(y)   # (0,0,0, ... ,1,1,1, ... ,2,2,2, ...)

# *** 머신러닝에서는 1차원으로 받아들여야 해서 원핫인코딩, 투카테고리칼 하지 않음
# y = to_categorical(y)
# ic(y[:5])
# [0,0,0,0,0]
# [[1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [1. 0. 0.]]
ic(y.shape)   # (150, )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# 1-2. 데이터 전처리
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


# 2. 모델(머신러닝에서는 정의만 해주면 됨)

allAlgorithms = all_estimators(type_filter='classifier')
# ic(allAlgorithms)
print('모델의 개수:',len(allAlgorithms))

for (name , algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train) 

        y_predict = model.predict(x_test) 
        acc = accuracy_score(y_test,y_predict)
        print(name,'의 정답률: ', acc)
    except:
        # continue
        print(name,'은 없는놈!!')
# predict는 100퍼센트 다 있음 가끔 score가 없는 경우도 있음
# try, except로 에러뜬거 무시하고 계속해서 정상적으로 출력


'''

# model = SVC()
# ic| acc: 0.9555555555555556

# model = KNeighborsClassifier()
# ic| acc: 0.8888888888888888

# model = LogisticRegression()
# ic| acc: 0.9777777777777777

# model = DecisionTreeClassifier()
# ic| acc: 0.9111111111111111

model = RandomForestClassifier()
# ic| acc: 0.8888888888888888

# model = LinearSVC()
# ic| acc: 0.9111111111111111


# 3. 훈련(컴파일 포함되어 있어서 컴파일 할 필요 없음)
model.fit(x_train, y_train)


# 4. 평가(evaluate 대신 score 사용함!!), 예측
results = model.score(x_test, y_test)       # accuracy
print('model.score:', results)

from sklearn.metrics import r2_score, accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
ic(acc)

ic(y_test[:5])
y_predict = model.predict(x_test[:5])
ic(y_predict)   # 소프트맥스 통과한 값


'''
