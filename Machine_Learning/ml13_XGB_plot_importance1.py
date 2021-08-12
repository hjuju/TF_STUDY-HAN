from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from xgboost.plotting import plot_importance
datasets = load_boston()

# 1. 데이터

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델
# model = DecisionTreeRegressor(max_depth=4)
model = XGBRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc:', acc)

print(model.feature_importances_)

import matplotlib.pyplot as plt
import numpy as np

# def plot_feature_importance_dataset(model): 
#     n_feature = datasets.data.shape[1]
#     plt.barh(np.arange(n_feature), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_feature), datasets.feature_names)
#     plt.xlabel("feature Importances")
#     plt.ylabel("features")
#     plt.ylim(-1, n_feature)

# plot_feature_importance_dataset(model)
# plt.show()

plot_importance(model)
plt.show()

'''
[0.01447935 0.00363372  0.01479119 0.00134153 0.06949984 0.30128643
 0.01220458 0.0518254  0.0175432  0.03041655 0.04246345 0.01203115
 0.42848358]
'''
'''
RandomForestRegressor
acc: 0.9202646740153897
[0.04008116 0.00137875 0.00692123 0.0010484  0.02420762 0.39384409
 0.01329015 0.06573008 0.0043216  0.01323685 0.01666752 0.01071775
 0.40855481]

DecisionTreeRegressor
acc: 0.8774175457631728
[0.03878833 0.         0.         0.         0.00765832 0.29639913
 0.         0.05991689 0.         0.01862509 0.         0.
 0.57861225]

'''