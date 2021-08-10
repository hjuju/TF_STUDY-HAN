import numpy as np
from sklearn.datasets import load_wine
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore') # 경고무시

### 머신러닝(evaluate -> score)

datasets = load_wine()

# 1. 데이터
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)  # (150, 4), (150,)->(150, 3)
ic(y.shape)   # (150, )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# 1-2. 데이터 전처리
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


# 2. 모델(머신러닝에서는 정의만 해주면 됨)

allAlgorithms = all_estimators(type_filter='classifier')
# ic(allAlgorithms)
print('모델의 개수:',len(allAlgorithms))

for (name , algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train) 

        y_predict = model.predict(x_test) 
        acc = accuracy_score(y_test,y_predict)
        print(name,'의 정답률: ', acc)
    except:
        # continue
        print(name,'은 없는놈!!')
# predict는 100퍼센트 다 있음 가끔 score가 없는 경우도 있음
# try, except로 에러뜬거 무시하고 계속해서 정상적으로 출력

'''
모델의 개수: 41
AdaBoostClassifier 의 정답률:  0.8888888888888888
BaggingClassifier 의 정답률:  1.0
BernoulliNB 의 정답률:  0.4166666666666667
CalibratedClassifierCV 의 정답률:  0.9444444444444444
CategoricalNB 은 없는놈!!
ClassifierChain 은 없는놈!!
ComplementNB 의 정답률:  0.6944444444444444
DecisionTreeClassifier 의 정답률:  0.9722222222222222
DummyClassifier 의 정답률:  0.4166666666666667
ExtraTreeClassifier 의 정답률:  0.9444444444444444
ExtraTreesClassifier 의 정답률:  1.0
GaussianNB 의 정답률:  1.0
GaussianProcessClassifier 의 정답률:  0.4444444444444444
GradientBoostingClassifier 의 정답률:  0.9722222222222222
HistGradientBoostingClassifier 의 정답률:  0.9722222222222222
KNeighborsClassifier 의 정답률:  0.6944444444444444
LabelPropagation 의 정답률:  0.5277777777777778
LabelSpreading 의 정답률:  0.5277777777777778
LinearDiscriminantAnalysis 의 정답률:  1.0
LinearSVC 의 정답률:  0.8333333333333334
LogisticRegression 의 정답률:  0.9722222222222222
LogisticRegressionCV 의 정답률:  0.9722222222222222
MLPClassifier 의 정답률:  0.2222222222222222
MultiOutputClassifier 은 없는놈!!
MultinomialNB 의 정답률:  0.7777777777777778
NearestCentroid 의 정답률:  0.6944444444444444
NuSVC 의 정답률:  0.9444444444444444
OneVsOneClassifier 은 없는놈!!
OneVsRestClassifier 은 없는놈!!
OutputCodeClassifier 은 없는놈!!
PassiveAggressiveClassifier 의 정답률:  0.6388888888888888
Perceptron 의 정답률:  0.6388888888888888
QuadraticDiscriminantAnalysis 의 정답률:  0.9722222222222222
RadiusNeighborsClassifier 은 없는놈!!
RandomForestClassifier 의 정답률:  1.0
RidgeClassifier 의 정답률:  1.0
RidgeClassifierCV 의 정답률:  1.0
SGDClassifier 의 정답률:  0.25
SVC 의 정답률:  0.6944444444444444
StackingClassifier 은 없는놈!!
VotingClassifier 은 없는놈!!
'''
