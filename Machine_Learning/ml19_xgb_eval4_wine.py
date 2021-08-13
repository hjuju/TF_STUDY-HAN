from math import gamma
from xgboost import XGBRegressor, XGBClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, RobustScaler, PowerTransformer
from icecream import ic


#1. 데이터

datasets = load_wine()
x = datasets['data']
y = datasets['target']

# ic(x.shape, y.shape) # x.shape: (506, 13), y.shape: (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#1-2. 데이터 전처리

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBClassifier(n_estimators=200, learning_rate=0.001, n_jobs=1)
# model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric=['merror', 'mlogloss'],
            eval_set=[(x_train, y_train),(x_test, y_test)]) # verbose를 출력하기 위해서는 평가하는 것이 무엇인지 명시해줘야함 / validation을 eval_set에 넣어줌

#4. 평가
result = model.score(x_test, y_test)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)

ic(result)
ic(r2)

'''
파라미터 기본값

ic| result: 1.0
ic| r2: 1.0

파라미터 튜닝

ic| result: 1.0
ic| r2: 1.0




'''