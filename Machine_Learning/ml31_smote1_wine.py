from imblearn.over_sampling import SMOTE
from sklearn import datasets
from sklearn.datasets import load_wine
import pandas as pd
from xgboost import XGBRFClassifier
from sklearn.model_selection import train_test_split
import time
import warnings

from xgboost.sklearn import XGBClassifier
warnings.filterwarnings('ignore')
from icecream import ic

datasets = load_wine()
x = datasets.data
y = datasets.target

# ic(x.shape, y.shape) # x.shape: (178, 13), y.shape: (178,)

# ic(pd.Series(y).value_counts())
'''
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
2, 2]

1    71
0    59
2    48
'''

# 증폭실습 위해 임의로 데이터 삭제

x_new = x[:-30]
y_new = y[:-30]

ic(x_new.shape, y_new.shape)  # (148, 13), y_new.shape: (148,)
ic(pd.Series(y_new).value_counts()) 

'''
1    71
0    59
2    18
'''

x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, train_size=0.75, shuffle=True, random_state=66, stratify=y_new)
# stratify: class 비율(ratio)을 train / validation에 유지(한 쪽에 쏠려서 분배되는 것을 방지합니다) 
# 이 옵션을 지정해 주지 않고 classification 문제를 다룬다면, 성능의 차이가 많이 날 수 있음

ic(pd.Series(y_train).value_counts()) 
'''
1    53
0    44 -> 53
2    14 -> 53
'''

model = XGBRFClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
ic(score) # score: 0.9459459459459459

######################################### smote 적용 #########################################
print("="*40,'smote 적용',"="*40)


smote = SMOTE(random_state=66)

x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train) # 적은 데이터들을 가장 큰 데이터에 맞춰서 증폭

ic(pd.Series(y_smote_train).value_counts())
'''
0    53
1    53
2    53
'''

ic(x_smote_train.shape, y_smote_train.shape) # (159, 13), y_smote_train.shape: (159,)

print("smote 전: ", x_train.shape, y_train.shape)
print("smote 후: ", x_smote_train.shape, y_smote_train.shape)
print("smote 전 레이블 값 분포: \n", pd.Series(y_train).value_counts())
print("smote 후 레이블 값 분포: \n", pd.Series(y_smote_train).value_counts())

model2 = XGBClassifier(n_jobs=-1)
model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score2 = model2.score(x_test, y_test)
ic(score2) # score2: 0.972972972972973



