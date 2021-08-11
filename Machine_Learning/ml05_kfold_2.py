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

### 머신러닝(evaluate -> score)

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

# 1. 데이터
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)  # (150, 4), (150,)->(150, 3)
ic(y)   # (0,0,0, ... ,1,1,1, ... ,2,2,2, ...)


from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_splits=5

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


# 2. 모델(머신러닝에서는 정의만 해주면 됨)

model = SVC()
# Acc:  [0.95833333 1.         0.95833333 1.         0.875     ] 평균 Acc: 0.9583

# model = KNeighborsClassifier()
# Acc:  [0.91666667 1.         0.95833333 1.         0.95833333] 평균 Acc: 0.9667

# model = LogisticRegression()
# Acc:  [0.95833333 1.         0.95833333 1.         0.91666667] 평균 Acc: 0.9667

# model = DecisionTreeClassifier()
# Acc:  [0.95833333 0.91666667 0.91666667 1.         0.875     ] 평균 Acc: 0.9333

# model = RandomForestClassifier()
# Acc:  [0.95833333 0.95833333 0.95833333 1.         0.875     ] 평균 Acc: 0.95

# model = LinearSVC()
# Acc:  [1.         0.95833333 0.95833333 1.         0.91666667] 평균 Acc: 0.9667

# 4. 평가(evaluate 대신 score 사용함!!), 예측
   
scores = cross_val_score(model, x_train, y_train ,cv=kfold) # 어떤 모델로 훈련시킬 것인지, 데이터, 여기까지 하면 fit과 스코어까지 모두 완료-> 한번에 다섯번 훈련 시켜서 스코어 출력
print('Acc: ', scores, '평균 Acc:', round(np.mean(scores),4))






