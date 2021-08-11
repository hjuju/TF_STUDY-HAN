import numpy as np
from sklearn.datasets import load_iris
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore') # 경고무시
from sklearn.model_selection import KFold, cross_val_score

### 머신러닝(evaluate -> score)

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

# 1. 데이터
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)  # (150, 4), (150,)->(150, 3)
ic(y)   # (0,0,0, ... ,1,1,1, ... ,2,2,2, ...)

# 2. 모델(머신러닝에서는 정의만 해주면 됨)

allAlgorithms = all_estimators(type_filter='classifier')
# ic(allAlgorithms)
print('모델의 개수:',len(allAlgorithms))


kfold = KFold(n_splits=5, shuffle=True, random_state=66)
for (name , algorithm) in allAlgorithms:
    try:
        model = algorithm()
        scores = cross_val_score(model, x, y, cv=kfold)
        
        print(name, 'Acc: ', scores, '평균 Acc:', round(np.mean(scores),4))
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
AdaBoostClassifier Acc:  [0.63333333 0.93333333 1.         0.9        0.96666667] 평균 Acc: 0.8867
BaggingClassifier Acc:  [0.93333333 0.93333333 1.         0.9        0.96666667] 평균 Acc: 0.9467
BernoulliNB Acc:  [0.3        0.33333333 0.3        0.23333333 0.3       ] 평균 Acc: 0.2933
CalibratedClassifierCV Acc:  [0.9        0.83333333 1.         0.86666667 0.96666667] 평균 Acc: 0.9133
CategoricalNB Acc:  [0.9        0.93333333 0.93333333 0.9        1.        ] 평균 Acc: 0.9333
ClassifierChain 은 없는놈!!
ComplementNB Acc:  [0.66666667 0.66666667 0.7        0.6        0.7       ] 평균 Acc: 0.6667
DecisionTreeClassifier Acc:  [0.96666667 0.96666667 1.         0.9        0.93333333] 평균 Acc: 0.9533
DummyClassifier Acc:  [0.3        0.33333333 0.3        0.23333333 0.3       ] 평균 Acc: 0.2933
ExtraTreeClassifier Acc:  [0.83333333 0.86666667 0.96666667 0.83333333 0.96666667] 평균 Acc: 0.8933
ExtraTreesClassifier Acc:  [0.93333333 0.96666667 1.         0.86666667 0.96666667] 평균 Acc: 0.9467
GaussianNB Acc:  [0.96666667 0.9        1.         0.9        0.96666667] 평균 Acc: 0.9467
GaussianProcessClassifier Acc:  [0.96666667 0.96666667 1.         0.9        0.96666667] 평균 Acc: 0.96
GradientBoostingClassifier Acc:  [0.93333333 0.96666667 1.         0.93333333 0.96666667] 평균 Acc: 0.96
HistGradientBoostingClassifier Acc:  [0.86666667 0.96666667 1.         0.9        0.96666667] 평균 Acc: 0.94
KNeighborsClassifier Acc:  [0.96666667 0.96666667 1.         0.9        0.96666667] 평균 Acc: 0.96
LabelPropagation Acc:  [0.93333333 1.         1.         0.9        0.96666667] 평균 Acc: 0.96
LabelSpreading Acc:  [0.93333333 1.         1.         0.9        0.96666667] 평균 Acc: 0.96
LinearDiscriminantAnalysis Acc:  [1.  1.  1.  0.9 1. ] 평균 Acc: 0.98
LinearSVC Acc:  [0.96666667 0.96666667 1.         0.9        1.        ] 평균 Acc: 0.9667
LogisticRegression Acc:  [1.         0.96666667 1.         0.9        0.96666667] 평균 Acc: 0.9667
LogisticRegressionCV Acc:  [1.         0.96666667 1.         0.9        1.        ] 평균 Acc: 0.9733
MLPClassifier Acc:  [0.96666667 0.96666667 1.         0.93333333 1.        ] 평균 Acc: 0.9733
MultiOutputClassifier 은 없는놈!!
MultinomialNB Acc:  [0.96666667 0.93333333 1.         0.93333333 1.        ] 평균 Acc: 0.9667
NearestCentroid Acc:  [0.93333333 0.9        0.96666667 0.9        0.96666667] 평균 Acc: 0.9333
NuSVC Acc:  [0.96666667 0.96666667 1.         0.93333333 1.        ] 평균 Acc: 0.9733
OneVsOneClassifier 은 없는놈!!
OneVsRestClassifier 은 없는놈!!
OutputCodeClassifier 은 없는놈!!
PassiveAggressiveClassifier Acc:  [0.76666667 0.76666667 1.         0.7        0.86666667] 평균 Acc: 0.82
Perceptron Acc:  [0.66666667 0.66666667 0.93333333 0.73333333 0.9       ] 평균 Acc: 0.78
QuadraticDiscriminantAnalysis Acc:  [1.         0.96666667 1.         0.93333333 1.        ] 평균 Acc: 0.98
RadiusNeighborsClassifier Acc:  [0.96666667 0.9        0.96666667 0.93333333 1.        ] 평균 Acc: 0.9533
RandomForestClassifier Acc:  [0.9        0.96666667 1.         0.9        0.96666667] 평균 Acc: 0.9467
RidgeClassifier Acc:  [0.86666667 0.8        0.93333333 0.7        0.9       ] 평균 Acc: 0.84
RidgeClassifierCV Acc:  [0.86666667 0.8        0.93333333 0.7        0.9       ] 평균 Acc: 0.84
SGDClassifier Acc:  [0.86666667 0.96666667 0.76666667 0.83333333 0.63333333] 평균 Acc: 0.8133
SVC Acc:  [0.96666667 0.96666667 1.         0.93333333 0.96666667] 평균 Acc: 0.9667
StackingClassifier 은 없는놈!!
VotingClassifier 은 없는놈!!
'''