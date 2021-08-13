from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from icecream import ic


#1. 데이터

datasets = load_boston()
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
model = XGBRegressor(n_estimators=20, learning_rate=0.05, n_jobs=1)

#3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse', 'logloss'],
            eval_set=[(x_train, y_train),(x_test, y_test)]) # verbose를 출력하기 위해서는 평가하는 것이 무엇인지 명시해줘야함 / validation을 eval_set에 넣어줌

#4. 평가
result = model.score(x_test, y_test)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)

ic(result)
ic(r2)

print("=================================================================")
results = model.evals_result()
ic(results) # XGB에서 제공 -> tensorflow의 history와 비슷

from matplotlib import pyplot

results = model.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')

ax.legend()
pyplot.ylabel('AUC')
pyplot.title('XGBoost AUC')
pyplot.show()

'''

ic| result: 0.9221188601856797
ic| r2: 0.9221188601856797

파라미터 튜닝
ic| result: 0.931302184876482
ic| r2: 0.931302184876482

n_estimators=200, learning_rate=0.1, n_jobs=1


'''