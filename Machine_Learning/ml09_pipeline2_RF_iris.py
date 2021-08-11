import numpy as np
from sklearn.datasets import load_iris
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

### 머신러닝(evaluate -> score)

datasets = load_iris()

# 1. 데이터
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)  # (150, 4), (150,)->(150, 3)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# 1-2. 데이터 전처리
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# 2. 모델(머신러닝에서는 정의만 해주면 됨)
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression # 로지스틱회귀는 분류모델(회귀모델 X)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline

model = make_pipeline(MinMaxScaler(), RandomForestClassifier()) # 스케일러, 모델

# model = LogisticRegression()
# 3. 훈련(컴파일 포함되어 있어서 컴파일 할 필요 없음)
model.fit(x_train, y_train)


# 4. 평가(evaluate 대신 score 사용함!!), 예측

# print('최적의 매개변수: ', model.best_estimator_) # cv를 통해 나온 값 / GridSearchCV를 통해서만 출력 가능
# print("best_score: ", model.best_score_)

model_score = model.score(x_test, y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
ic(model_score)
ic(acc)

'''
RF
ic| model_score: 0.9
ic| acc: 0.9
'''

