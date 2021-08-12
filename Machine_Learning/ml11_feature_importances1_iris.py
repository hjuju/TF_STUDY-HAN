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