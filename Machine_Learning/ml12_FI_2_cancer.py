# 하위 20~25% 컬럼 제거하여 데이터셋 재구성 후 
# 각 모델별로 돌려서 결과 도출
# 기존 모델과 비교



from scipy.sparse import data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from icecream import ic
import math

datasets = load_breast_cancer()

# 1. 데이터

x = datasets.data
y = datasets.target



# df_x = pd.DataFrame(x).drop(([17, 19, 26, 16, 24, 10]), axis=1)

# # print(df_x)

# x = df_x.to_numpy()

# ic(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

# 2. 모델
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
model = GradientBoostingClassifier()
# model = XGBClassifier()

# 3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)


# acc = np.sort(acc, axis=1)
print('acc:', acc)

print(model.feature_importances_)

for i in model.feature_importances_:
    result = np.around(i / acc, 3)
    print(result)


fi_idx = np.argsort(model.feature_importances_)
fi_sort = np.sort(model.feature_importances_)

ic(fi_idx)
ic(fi_sort)


a = np.array(datasets.feature_names).reshape(1, datasets.data.shape[1])
b = model.feature_importances_.reshape(1, datasets.data.shape[1])
c = np.concatenate([a, b], axis=0)
d = np.transpose(c)
e = pd.DataFrame(d)
e.columns = ['Feature', 'Feature Importances']
print(e.sort_values(by=['Feature Importances'], axis=0, ascending=False).head(math.ceil(datasets.data.shape[1] * 0.7)))





'''
# XGBClassifier


# GradientBoostingClassifier

acc: 0.9649122807017544
[2.86060970e-04 4.57200861e-02 5.15346248e-04 1.73721555e-03
 2.35025489e-03 4.62393657e-03 1.81065087e-03 3.14979960e-02
 9.20704901e-04 0.00000000e+00 6.29310075e-03 5.38177356e-05
 2.26866131e-04 1.90204425e-02 7.81708611e-04 5.34895047e-03
 1.63847200e-03 1.46960501e-03 3.57946018e-06 7.51383213e-04
 2.87237992e-01 3.15914325e-02 2.67268175e-02 4.23703821e-01
 3.75209187e-03 4.79395217e-05 2.86796407e-03 9.85413148e-02
 5.84589428e-05 4.21989713e-04]


acc: 0.9590643274853801
[2.14619933e-05 4.62544283e-02 4.30464549e-04 2.34208521e-03
 7.39370489e-04 2.03301559e-03 2.88164980e-03 4.09394715e-02
 1.47057471e-03 2.21546095e-05 1.14026837e-04 1.41037647e-03
 2.36823776e-02 5.16359748e-04 6.22983982e-03 1.17206790e-04
 3.11039567e-01 3.26818418e-02 5.62900423e-03 4.18638481e-01
 3.07498239e-04 1.01557048e-01 9.11274307e-04 3.04208401e-05]


# RandomForestClassifier

acc: 0.9649122807017544
[0.02509449 0.01573871 0.03742911 0.04024552 0.00624973 0.00669752
 0.04050211 0.04980123 0.00375484 0.00576751 0.02804252 0.00511585
 0.0113979  0.03679616 0.00453034 0.0046066  0.00648392 0.00458868
 0.00495098 0.0072163  0.12642145 0.02049033 0.14955364 0.18680485
 0.00896834 0.01768038 0.01840243 0.11202297 0.00663664 0.00800893]

acc: 0.9707602339181286
[0.04944506 0.02244727 0.0347572  0.03346563 0.11727346 0.04845668
 0.15239266 0.03451025 0.22062534 0.17390188 0.03261341 0.08011116]

# DecisionTreeClassifier

acc: 0.935672514619883
[0.         0.02869086 0.         0.         0.         0.
 0.         0.00845356 0.         0.03043354 0.01481385 0.
 0.         0.         0.01502854 0.         0.         0.
 0.         0.         0.         0.01859782 0.         0.77570942
 0.         0.         0.         0.10827241 0.         0.        ]

acc: 0.9122807017543859
[0.83931746 0.16068254]
0.92
0.176

'''

import matplotlib.pyplot as plt

# feature_importances_가 지원되는 모델 시각화
def plot_feature_importance_dataset(model): 
    n_feature = datasets.data.shape[-1]
    plt.barh(np.arange(n_feature), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_feature), datasets.feature_names)
    plt.xlabel("feature Importances")
    plt.ylabel("features")
    plt.ylim(-1, n_feature)

plot_feature_importance_dataset(model)
plt.show()




