import numpy as np
from sklearn.datasets import load_iris
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression # 로지스틱회귀는 분류모델(회귀모델 X)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore') # 경고무시
from sklearn.metrics import accuracy_score

### 머신러닝(evaluate -> score)

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

# 1. 데이터
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)  # (150, 4), (150,)->(150, 3)

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_splits=5

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


# 2. 모델(머신러닝에서는 정의만 해주면 됨)



# model = SVC()
# Acc:  [0.96666667 0.96666667 1.         0.93333333 0.96666667] 평균 Acc: 0.9667

# model = KNeighborsClassifier()
# Acc:  [0.96666667 0.96666667 1.         0.9        0.96666667] 평균 Acc: 0.96

# model = LogisticRegression()
# Acc:  [1.         0.96666667 1.         0.9        0.96666667] 평균 Acc: 0.9667

# model = DecisionTreeClassifier()
# Acc:  [0.96666667 0.96666667 1.         0.9        0.93333333] 평균 Acc: 0.9533

# model = RandomForestClassifier()
# Acc:  [0.96666667 0.93333333 1.         0.86666667 0.96666667] 평균 Acc: 0.9467

# model = LinearSVC()
# Acc:  [0.96666667 0.96666667 1.         0.9        1.        ] 평균 Acc: 0.9667

parameter = [{"C":[1, 10, 100, 1000], "kernel":['linear']}, # 4 * 5(kfold숫자만큼) -> 20번 돌아감
             {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]}, # 3 * 1 * 3 * 5 -> 30번
             {"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}] # 4 * 1 * 2 * 5 = 40번 => 총 90번 모델이 돌아감


model = GridSearchCV(SVC(), parameter, cv=kfold) # 사용할 모델, parameter(정의), cv 명시 /  텐서플로로 하면 모델에 텐서플로 모델이 들어가면 됨

model.fit(x_train,y_train)

print('최적의 매개변수: ', model.best_estimator_) # cv를 통해 나온 값
print("best_score: ", model.best_score_)

y_predict = model.predict(x_test)
print("정답률: ", accuracy_score(y_test, y_predict))
# 위아래 녀석들은 같은 녀석들
print("model.score: ", model.score(x_test, y_test))

# 최적의 매개변수:  SVC(C=1, kernel='linear')
# best_score:  0.9916666666666668
# 정답률:  0.9666666666666667
# model.score:  0.9666666666666667


# 4. 평가(evaluate 대신 score 사용함!!), 예측

# scores = cross_val_score(model, x,y,cv=kfold) # 어떤 모델로 훈련시킬 것인지, 데이터, 여기까지 하면 fit과 스코어까지 모두 완료-> 한번에 다섯번 훈련 시켜서 스코어 출력
# print('Acc: ', scores, '평균 Acc:', round(np.mean(scores),4))






