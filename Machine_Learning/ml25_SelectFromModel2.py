# 실습 
# 데이터는 diabets 
#1. 상단모델에 그리드서치 또는 랜덤서치로 튜닝한 모델 구성
# 최적의 R2값과 피처임포턴스 

#2. 위 스레드값으로 selectfrommodel 돌려서 최적의 피처 개수 구할 것

#3. 위 피처개수로 조정한 뒤 다시 랜덤서치 또는 그리드서치 해서 최적의 R2구할 것

# 1번값과 3번값 비교

# 실습

# 모델: RandomForestClassifier

# 실습
# m07_1 최적의 파라미터 값을 가지고 model 구성 결과 도출

import numpy as np
from sklearn.datasets import load_diabetes
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression # 로지스틱회귀는 분류모델(회귀모델 X)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore') # 경고무시
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.feature_selection import SelectFromModel

### 머신러닝(evaluate -> score)

datasets = load_diabetes()

# 1. 데이터
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)  # (150, 4), (150,)->(150, 3)

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# 2. 모델(머신러닝에서는 정의만 해주면 됨)

n_splits=5

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

md = 'RR__'
parameter = [ {f'{md}n_jobs':[-1], f'{md}n_estimators':[1, 10, 100], f'{md}max_depth':[6,8,10,12], f'{md}min_samples_leaf':[8,12,18], f'{md}min_samples_split':[8,16,20]},
              {f'{md}n_jobs':[-1], f'{md}n_estimators':[2, 30, 200],f'{md}max_depth':[10,16], f'{md}min_samples_leaf':[10,14], f'{md}min_samples_split':[4,10,30]},
              {f'{md}n_jobs':[-1], f'{md}n_estimators':[50, 80],f'{md}max_depth':[10, 12], f'{md}min_samples_leaf':[12,18], f'{md}min_samples_split':[16,20], f'{md}criterion':['entropy', 'gini']}
]
              
pipe = Pipeline([("scaler", MinMaxScaler()), ("RR", XGBRegressor())])

model = RandomizedSearchCV(pipe, parameter, cv=kfold, verbose=1) 

model.fit(x_train,y_train)

print('최적의 매개변수: ', model.best_estimator_) # cv를 통해 나온 값 / GridSearchCV를 통해서만 출력 가능
print("best_score: ", model.best_params_)
print("best_score: ", model.best_score_)

y_predict = model.predict(x_test)
R2= r2_score(y_test, y_predict)
ic(R2)

'''
best_score:  {'RR__n_jobs': -1, 'RR__n_estimators': 100, 'RR__min_samples_split': 8, 'RR__min_samples_leaf': 12, 'RR__max_depth': 10}
best_score:  0.4881297099491132
ic| R2: 0.41314001032640535
'''
model = XGBRegressor(n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)

threshold = np.sort(model.feature_importances_) # 순서 정렬

threshold = threshold[0:6]

for thresh in threshold:
    selection = SelectFromModel(model, threshold=thresh, prefit=True) 
    # thresh 미만의 컬럼들은 하나씩 갱신하여 삭제 됨 / 순서대로 0번째, 1번째... 컬럼을 삭제하고, 13개, 12개 줄여가면서 모델 구성
    # ic(selection)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    ic(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100))












