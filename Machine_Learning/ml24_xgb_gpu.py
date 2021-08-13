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
model = XGBRegressor(n_estimators=10000, learning_rate=0.05, n_jobs=2, tree_method='gpu_hist', 
                        predictor='gpu_predictor',
                        gpu_id=0) 
                        # gpu_id=여러개 GPU사용 시 선택 가능
                        # gpu / cpu_predictor -> predict를 무엇으로 할 지 선택

import time
start = time.time()
#3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric='rmse',
            eval_set=[(x_train, y_train),(x_test, y_test)]) # early_stopping_rounds은 몇번 갱신이 없으면 멈출건지 설정

print('걸린시간: ', time.time() - start)


# GPU가 빠른지 CPU가 빠른지 확인 해보고 돌려야함 데이터에 따라서 CPU가 빠를 수도 있음

'''
n_jobs=1

걸린시간:  5.956038475036621

n_jobs=2

걸린시간:  5.16814661026001

n_jobs=4
걸린시간:  4.7991623878479

n_jobs=8
걸린시간:  4.989650011062622

GPU사용
걸린시간:  27.843310594558716
'''
