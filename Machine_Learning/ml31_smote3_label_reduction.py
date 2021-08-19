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
from sklearn.preprocessing import StandardScaler

datasets = pd.read_csv('../_data/wine/winequality-white.csv', index_col=None, header=0, sep=';')

datasets = datasets.values

x = datasets[:,:11]
y = datasets[:,11]

# ic(x.shape, y.shape) # x.shape: (178, 13), y.shape: (178,)

ic(pd.Series(y).value_counts())

'''
6.0    2198
5.0    1457
7.0     880
8.0     175
4.0     163
3.0      20
9.0       5
'''

#############################################
##### 라벨 통합
#############################################

print("="*50)

for index, value in enumerate(y): # 인덱스와 밸류 반환
    if value == 9 :
        y[index] = 8
    elif value == 8:
        y[index] = 8
    elif value == 7:
        y[index] = 6
    elif value == 6:
        y[index] = 6
    elif value == 5:
        y[index] = 5
    elif value == 4:
        y[index] = 5
    elif value == 3:
        y[index] = 5
    
   
ic(pd.Series(y).value_counts())

'''
6.0    2198
5.0    1457
7.0     880
8.0     180
4.0     163
3.0      20
'''

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=98, stratify=y)
# stratify: class 비율(ratio)을 train / validation에 유지(한 쪽에 쏠려서 분배되는 것을 방지합니다) 
# 이 옵션을 지정해 주지 않고 classification 문제를 다룬다면, 성능의 차이가 많이 날 수 있음

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

ic(pd.Series(y_train).value_counts()) 

'''
6.0    1648
5.0    1093
7.0     660
8.0     131
4.0     122
3.0      15
9.0       4
'''

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
ic(score) # score: 0.6587755102040816

######################################### smote 적용 #########################################
print("="*40,'smote 적용',"="*40)

smote = SMOTE(random_state=77) # k_neighbors가 적어지면 성능이 떨어짐

start = time.time()
x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train) # 적은 데이터들을 가장 큰 데이터에 맞춰서 증폭
end = time.time() - start

print("걸린시간:", end)
ic(pd.Series(y_smote_train).value_counts())

'''
6.0    1648
5.0    1648
4.0    1648
9.0    1648
8.0    1648
7.0    1648
3.0    1648
'''

ic(x_smote_train.shape, y_smote_train.shape) # (159, 13), y_smote_train.shape: (159,)

print("smote 전: ", x_train.shape, y_train.shape)
print("smote 후: ", x_smote_train.shape, y_smote_train.shape)
print("smote 전 레이블 값 분포: \n", pd.Series(y_train).value_counts())
print("smote 후 레이블 값 분포: \n", pd.Series(y_smote_train).value_counts())

model2 = XGBClassifier(n_jobs=-1)

model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')



score2 = model2.score(x_test, y_test)
ic(score2) 
'''
9 -> 8 score2: 0.6334693877551021

9 -> 8, 3 -> 4 0.626938775510204

0.6563265306122449

'''


