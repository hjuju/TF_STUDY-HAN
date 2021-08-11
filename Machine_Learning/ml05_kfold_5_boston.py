import numpy as np
from sklearn.datasets import load_boston
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression # 로지스틱회귀는 분류모델(회귀모델 X)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore') # 경고무시

### 머신러닝(evaluate -> score)

datasets = load_boston()
print(datasets.DESCR)
print(datasets.feature_names)

# 1. 데이터
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)  # (150, 4), (150,)->(150, 3)

from sklearn.model_selection import train_test_split, KFold, cross_val_score
n_splits=5

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


# 2. 모델(머신러닝에서는 정의만 해주면 됨)



# model = SVC()
# 

# model = KNeighborsRegressor()
# Acc:  [0.59008727 0.68112533 0.55680192 0.4032667  0.41180856] 평균 Acc: 0.5286

# model = LogisticRegression()
# 

# model = DecisionTreeRegressor()
# Acc:  [0.80042366 0.60706893 0.75970398 0.73315285 0.80154249] 평균 Acc: 0.7404

model = RandomForestRegressor()
# Acc:  [0.92172052 0.84917565 0.82002461 0.88876168 0.90432548] 평균 Acc: 0.8768

# model = LinearSVC()
# Acc:  [0.91666667 0.91666667 0.77777778 0.82857143 0.91428571] 평균 Acc: 0.8708


# 4. 평가(evaluate 대신 score 사용함!!), 예측

scores = cross_val_score(model, x,y,cv=kfold) # 어떤 모델로 훈련시킬 것인지, 데이터, 여기까지 하면 fit과 스코어까지 모두 완료-> 한번에 다섯번 훈련 시켜서 스코어 출력
print('Acc: ', scores, '평균 Acc:', round(np.mean(scores),4))






