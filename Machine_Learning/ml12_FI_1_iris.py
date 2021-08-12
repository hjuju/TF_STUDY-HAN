# 하위 20~25% 컬럼 제거하여 데이터셋 재구성 후 
# 각 모델별로 돌려서 결과 도출
# 기존 모델과 비교



from scipy.sparse import data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from icecream import ic

datasets = load_iris()

# 1. 데이터

x = datasets.data
y = datasets.target

df_x = pd.DataFrame(x).drop(([0,1,3]), axis=1)

print(df_x)

x = df_x.to_numpy()

ic(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

# 2. 모델
model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = XGBClassifier()

# 3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc:', acc)

print(model.feature_importances_)

for i in model.feature_importances_:
    rank = np.around(i / acc, 3)
    print(rank)


'''
# XGBClassifier
acc: 0.9111111111111111

acc: 0.9111111111111111
[0.73307    0.26692995]
0.80459
0.29297

# GradientBoostingClassifier
acc: 0.8888888888888888
[0.00183217 0.01246847 0.49108757 0.49461179]
0.00206
0.01403
0.55247
0.55644


acc: 0.9111111111111111
[0.52207286 0.47792714]
0.57301
0.52455

# RandomForestClassifier

acc: 0.9111111111111111
[0.11324069 0.04733668 0.38733681 0.45208582]
0.124
0.052
0.425
0.496

acc: 0.9333333333333333
[0.57223421 0.42776579]
0.613
0.458

# DecisionTreeClassifier

acc: 0.9111111111111111
[0.         0.01906837 0.91835184 0.06257979]
0.0
0.021
1.008
0.069

acc: 0.9111111111111111
[1.]
1.098

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




