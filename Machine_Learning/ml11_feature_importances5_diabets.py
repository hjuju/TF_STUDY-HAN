from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
datasets = load_diabetes()

# 1. 데이터

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. 모델
# model = DecisionTreeRegressor(max_depth=10)
model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc:', acc)

print(model.feature_importances_)

'''
RandomForestRegressor
acc: 0.3514217454108447
[0.06019095 0.01075585 0.27585365 0.1099317  0.04475429 0.05587376
 0.04820403 0.02188837 0.30248881 0.07005859]

DecisionTreeRegressor
acc: -0.19533497950254386
[0.06727573 0.00969423 0.22130172 0.11620656 0.03493696 0.05579832
 0.0441194  0.00411934 0.36908064 0.07746711]
 
'''