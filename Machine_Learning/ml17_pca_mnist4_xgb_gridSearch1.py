# 0.999 이상의 n_componet=?를 사용하여 xgb 만들것 


import numpy as np
from tensorflow.keras.datasets import mnist
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression # 로지스틱회귀는 분류모델(회귀모델 X)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
from xgboost.sklearn import XGBClassifier
warnings.filterwarnings('ignore') # 경고무시
from sklearn.metrics import accuracy_score


parameter = [{"n_estimators":[100, 200, 300], "learning_rate":[0.1,0.3,0.001,0.01], "max_depth":[4,5,6]},
              {"n_estimators":[90, 100, 110], "learning_rate":[0.1,0.001,0.01], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
              {"n_estimators":[90, 110], "learning_rate":[0.1,0.001,0.5], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1],"colsample_bylevel":[0.6,0.7,0.9]}]


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1. 데이터
x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)
ic(x.shape, y.shape)  # (150, 4), (150,)->(150, 3)

# ic| x.shape: (70000, 28, 28), y.shape: (70000,)



from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)



# # 2. 모델(머신러닝에서는 정의만 해주면 됨)

# n_splits=5

# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

 
# model = GridSearchCV(XGBClassifier(), parameter, cv=kfold) # 사용할 모델, parameter(정의), cv 명시 /  텐서플로로 하면 모델에 텐서플로 모델이 들어가면 됨
# # model = SVC(C=1, kernel='linear')

# model.fit(x_train,y_train)

# print('최적의 매개변수: ', model.best_estimator_) # cv를 통해 나온 값 / GridSearchCV를 통해서만 출력 가능
# print("best_score: ", model.best_score_)








