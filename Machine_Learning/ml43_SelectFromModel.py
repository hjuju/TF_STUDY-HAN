from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from icecream import ic
from sklearn.feature_selection import SelectFromModel

#1. 데이터
# x,y = load_boston()
# x = datasets.data
# y = datasets.target

x,y = load_boston(return_X_y=True)
ic(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)


#2. 모델
model = XGBRegressor(n_jobs=8)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)
# ic(score)

threshold = np.sort(model.feature_importances_) # 순서 정렬
ic(threshold) # 몇 개의 컬럼을 삭제하고 학습을 시켰을 때 성능이 더 좋을 수도 있음

'''
[0.00134153, 0.00363372, 0.01203115, 0.01220458, 0.01447935,
                0.01479119, 0.0175432 , 0.03041655, 0.04246345, 0.0518254 ,
                0.06949984, 0.30128643, 0.42848358]
'''


# 피쳐 삭제 피쳐를 삭제하고, PCA를 통해 압축하기도 가능
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

