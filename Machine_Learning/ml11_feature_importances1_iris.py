from scipy.sparse import data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
datasets = load_iris()

# 1. 데이터

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

#2. 모델
# model = DecisionTreeClassifier(max_depth=4)
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc:', acc)

print(model.feature_importances_)

# [0.         0.0125026  0.03213177 0.95536562] # 첫번째 칼럼은 영향을 주고있지 않다. 삭제해도 됨

import matplotlib.pyplot as plt
import numpy as np


#  feature_importances_가 지원되는 모델 시각화
def plot_feature_importance_dataset(model): 
    n_feature = datasets.data.shape[1]
    plt.barh(np.arange(n_feature), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_feature), datasets.feature_names)
    plt.xlabel("feature Importances")
    plt.ylabel("features")
    plt.ylim(-1, n_feature)

plot_feature_importance_dataset(model)
plt.show()