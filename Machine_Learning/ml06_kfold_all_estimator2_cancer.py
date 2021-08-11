import numpy as np
from sklearn.datasets import load_breast_cancer
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore') # 경고무시
from sklearn.model_selection import KFold, cross_val_score

### 머신러닝(evaluate -> score)

datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)

# 1. 데이터
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)  # (150, 4), (150,)->(150, 3)

# 2. 모델(머신러닝에서는 정의만 해주면 됨)

allAlgorithms = all_estimators(type_filter='classifier')
# ic(allAlgorithms)
print('모델의 개수:',len(allAlgorithms))


kfold = KFold(n_splits=5, shuffle=True, random_state=66)
for (name , algorithm) in allAlgorithms:
    try:
        model = algorithm()
        scores = cross_val_score(model, x, y, cv=kfold)
        
        print('name: ', name, 'Acc: ', scores, '평균 Acc:', round(np.mean(scores),4))
    except:
        # continue
        print(name,'은 없는놈!!')
# predict는 100퍼센트 다 있음 가끔 score가 없는 경우도 있음
# try, except로 에러뜬거 무시하고 계속해서 정상적으로 출력

'''
모델의 개수: 41
AdaBoostClassifier 의 정답률:  0.6333333333333333
BaggingClassifier 의 정답률:  0.9666666666666667
BernoulliNB 의 정답률:  0.3
CalibratedClassifierCV 의 정답률:  0.9
CategoricalNB 의 정답률:  0.9
ClassifierChain 은 없는놈!!
ComplementNB 의 정답률:  0.6666666666666666
DecisionTreeClassifier 의 정답률:  0.9333333333333333
DummyClassifier 의 정답률:  0.3
ExtraTreeClassifier 의 정답률:  0.9666666666666667
ExtraTreesClassifier 의 정답률:  0.9333333333333333
GaussianNB 의 정답률:  0.9666666666666667
GaussianProcessClassifier 의 정답률:  0.9666666666666667
GradientBoostingClassifier 의 정답률:  0.9666666666666667
HistGradientBoostingClassifier 의 정답률:  0.8666666666666667
KNeighborsClassifier 의 정답률:  0.9666666666666667
LabelPropagation 의 정답률:  0.9333333333333333
LabelSpreading 의 정답률:  0.9333333333333333
LinearDiscriminantAnalysis 의 정답률:  1.0
LinearSVC 의 정답률:  0.9666666666666667
LogisticRegression 의 정답률:  1.0
LogisticRegressionCV 의 정답률:  1.0
MLPClassifier 의 정답률:  0.9666666666666667
MultiOutputClassifier 은 없는놈!!
MultinomialNB 의 정답률:  0.9666666666666667
NearestCentroid 의 정답률:  0.9333333333333333
NuSVC 의 정답률:  0.9666666666666667
OneVsOneClassifier 은 없는놈!!
OneVsRestClassifier 은 없는놈!!
OutputCodeClassifier 은 없는놈!!
PassiveAggressiveClassifier 의 정답률:  0.7
Perceptron 의 정답률:  0.9333333333333333
QuadraticDiscriminantAnalysis 의 정답률:  1.0
RadiusNeighborsClassifier 의 정답률:  0.9666666666666667
RandomForestClassifier 의 정답률:  0.9666666666666667
RidgeClassifier 의 정답률:  0.8666666666666667
RidgeClassifierCV 의 정답률:  0.8666666666666667
SGDClassifier 의 정답률:  0.6666666666666666
SVC 의 정답률:  0.9666666666666667
StackingClassifier 은 없는놈!!
VotingClassifier 은 없는놈!!

K_FOLD
모델의 개수: 41
name:  AdaBoostClassifier Acc:  [0.94736842 0.99122807 0.94736842 0.96491228 0.97345133] 평균 Acc: 0.9649
name:  BaggingClassifier Acc:  [0.93859649 0.92105263 0.94736842 0.94736842 0.94690265] 평균 Acc: 0.9403
name:  BernoulliNB Acc:  [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858] 평균 Acc: 0.6274
name:  CalibratedClassifierCV Acc:  [0.89473684 0.93859649 0.89473684 0.92982456 0.97345133] 평균 Acc: 0.9263
name:  CategoricalNB Acc:  [nan nan nan nan nan] 평균 Acc: nan
ClassifierChain 은 없는놈!!
name:  ComplementNB Acc:  [0.86842105 0.92982456 0.87719298 0.9122807  0.89380531] 평균 Acc: 0.8963
name:  DecisionTreeClassifier Acc:  [0.93859649 0.9122807  0.9122807  0.9122807  0.95575221] 평균 Acc: 0.9262
name:  DummyClassifier Acc:  [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858] 평균 Acc: 0.6274
name:  ExtraTreeClassifier Acc:  [0.93859649 0.94736842 0.90350877 0.9122807  0.87610619] 평균 Acc: 0.9156
name:  ExtraTreesClassifier Acc:  [0.96491228 0.97368421 0.96491228 0.95614035 0.99115044] 평균 Acc: 0.9702
name:  GaussianNB Acc:  [0.93859649 0.96491228 0.9122807  0.93859649 0.95575221] 평균 Acc: 0.942
name:  GaussianProcessClassifier Acc:  [0.87719298 0.89473684 0.89473684 0.94736842 0.94690265] 평균 Acc: 0.9122
name:  GradientBoostingClassifier Acc:  [0.95614035 0.96491228 0.95614035 0.93859649 0.98230088] 평균 Acc: 0.9596
name:  HistGradientBoostingClassifier Acc:  [0.97368421 0.98245614 0.96491228 0.96491228 0.98230088] 평균 Acc: 0.9737
name:  KNeighborsClassifier Acc:  [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221] 평균 Acc: 0.928
name:  LabelPropagation Acc:  [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053] 평균 Acc: 0.3902
name:  LabelSpreading Acc:  [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053] 평균 Acc: 0.3902
name:  LinearDiscriminantAnalysis Acc:  [0.94736842 0.98245614 0.94736842 0.95614035 0.97345133] 평균 Acc: 0.9614
name:  LinearSVC Acc:  [0.85087719 0.93859649 0.90350877 0.92982456 0.87610619] 평균 Acc: 0.8998
name:  LogisticRegression Acc:  [0.93859649 0.95614035 0.88596491 0.94736842 0.96460177] 평균 Acc: 0.9385
name:  LogisticRegressionCV Acc:  [0.96491228 0.97368421 0.92105263 0.96491228 0.96460177] 평균 Acc: 0.9578
name:  MLPClassifier Acc:  [0.9122807  0.92982456 0.86842105 0.94736842 0.94690265] 평균 Acc: 0.921
MultiOutputClassifier 은 없는놈!!
name:  MultinomialNB Acc:  [0.85964912 0.92105263 0.87719298 0.9122807  0.89380531] 평균 Acc: 0.8928
name:  NearestCentroid Acc:  [0.86842105 0.89473684 0.85964912 0.9122807  0.91150442] 평균 Acc: 0.8893
name:  NuSVC Acc:  [0.85964912 0.9122807  0.83333333 0.87719298 0.88495575] 평균 Acc: 0.8735
OneVsOneClassifier 은 없는놈!!
OneVsRestClassifier 은 없는놈!!
OutputCodeClassifier 은 없는놈!!
name:  PassiveAggressiveClassifier Acc:  [0.85964912 0.93859649 0.90350877 0.92982456 0.71681416] 평균 Acc: 0.8697
name:  Perceptron Acc:  [0.40350877 0.80701754 0.85964912 0.86842105 0.94690265] 평균 Acc: 0.7771
name:  QuadraticDiscriminantAnalysis Acc:  [0.93859649 0.95614035 0.93859649 0.98245614 0.94690265] 평균 Acc: 0.9525
name:  RadiusNeighborsClassifier Acc:  [nan nan nan nan nan] 평균 Acc: nan
name:  RandomForestClassifier Acc:  [0.97368421 0.95614035 0.96491228 0.94736842 0.97345133] 평균 Acc: 0.9631
name:  RidgeClassifier Acc:  [0.95614035 0.98245614 0.92105263 0.95614035 0.95575221] 평균 Acc: 0.9543
name:  RidgeClassifierCV Acc:  [0.94736842 0.97368421 0.93859649 0.95614035 0.96460177] 평균 Acc: 0.9561
name:  SGDClassifier Acc:  [0.75438596 0.92982456 0.85964912 0.66666667 0.95575221] 평균 Acc: 0.8333
name:  SVC Acc:  [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177] 평균 Acc: 0.921
StackingClassifier 은 없는놈!!
VotingClassifier 은 없는놈!!
'''