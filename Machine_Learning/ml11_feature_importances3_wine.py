from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
datasets = load_wine()

# 1. 데이터

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

#2. 모델
# model = DecisionTreeClassifier(max_depth=3)
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc:', acc)

print(model.feature_importances_)


'''
Randomforest
acc: 1.0
[0.13638494 0.02695159 0.01992916 0.03449203 0.02784648 0.05486211
 0.18406798 0.012637   0.01301224 0.13307032 0.08983627 0.11639688
 0.150513  ]

DecisionTreeClassifier
acc: 0.9629629629629629
[0.00586196 0.         0.         0.         0.         0.
 0.03189664 0.         0.         0.1325059  0.         0.39349299
 0.43624251]
'''