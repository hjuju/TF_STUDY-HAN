import numpy as np
from sklearn.datasets import load_wine
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore') # 경고무시
from sklearn.model_selection import KFold, cross_val_score

### 머신러닝(evaluate -> score)

datasets = load_wine()
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
AdaBoostClassifier Acc:  [0.88888889 0.86111111 0.88888889 0.94285714 0.97142857] 평균 Acc: 0.9106
BaggingClassifier Acc:  [1.         0.91666667 0.86111111 0.97142857 0.97142857] 평균 Acc: 0.9441
BernoulliNB Acc:  [0.41666667 0.47222222 0.27777778 0.48571429 0.34285714] 평균 Acc: 0.399
CalibratedClassifierCV Acc:  [0.94444444 0.94444444 0.88888889 0.88571429 0.91428571] 평균 Acc: 0.9156
CategoricalNB Acc:  [       nan        nan        nan 0.94285714        nan] 평균 Acc: nan
ClassifierChain 은 없는놈!!
ComplementNB Acc:  [0.69444444 0.80555556 0.55555556 0.6        0.6       ] 평균 Acc: 0.6511
DecisionTreeClassifier Acc:  [0.91666667 0.97222222 0.91666667 0.88571429 0.94285714] 평균 Acc: 0.9268
DummyClassifier Acc:  [0.41666667 0.47222222 0.27777778 0.48571429 0.34285714] 평균 Acc: 0.399
ExtraTreeClassifier Acc:  [0.88888889 0.86111111 0.83333333 1.         0.97142857] 평균 Acc: 0.911
ExtraTreesClassifier Acc:  [1.         0.97222222 1.         0.97142857 1.        ] 평균 Acc: 0.9887
GaussianNB Acc:  [1.         0.91666667 0.97222222 0.97142857 1.        ] 평균 Acc: 0.9721
GaussianProcessClassifier Acc:  [0.44444444 0.30555556 0.55555556 0.62857143 0.45714286] 평균 Acc: 0.4783
GradientBoostingClassifier Acc:  [0.97222222 0.91666667 0.88888889 0.97142857 0.97142857] 평균 Acc: 0.9441
HistGradientBoostingClassifier Acc:  [0.97222222 0.94444444 1.         0.97142857 1.        ] 평균 Acc: 0.9776
KNeighborsClassifier Acc:  [0.69444444 0.77777778 0.61111111 0.62857143 0.74285714] 평균 Acc: 0.691
LabelPropagation Acc:  [0.52777778 0.47222222 0.5        0.4        0.54285714] 평균 Acc: 0.4886
LabelSpreading Acc:  [0.52777778 0.47222222 0.5        0.4        0.54285714] 평균 Acc: 0.4886
LinearDiscriminantAnalysis Acc:  [1.         0.97222222 1.         0.97142857 1.        ] 평균 Acc: 0.9887
LinearSVC Acc:  [0.94444444 0.80555556 0.72222222 0.85714286 0.91428571] 평균 Acc: 0.8487
LogisticRegression Acc:  [0.97222222 0.94444444 0.94444444 0.94285714 1.        ] 평균 Acc: 0.9608
LogisticRegressionCV Acc:  [1.         0.94444444 0.97222222 0.94285714 0.97142857] 평균 Acc: 0.9662
MLPClassifier Acc:  [0.80555556 0.91666667 0.88888889 0.05714286 0.85714286] 평균 Acc: 0.7051
MultiOutputClassifier 은 없는놈!!
MultinomialNB Acc:  [0.77777778 0.91666667 0.86111111 0.82857143 0.82857143] 평균 Acc: 0.8425
NearestCentroid Acc:  [0.69444444 0.72222222 0.69444444 0.77142857 0.74285714] 평균 Acc: 0.7251
NuSVC Acc:  [0.91666667 0.86111111 0.91666667 0.85714286 0.8       ] 평균 Acc: 0.8703
OneVsOneClassifier 은 없는놈!!
OneVsRestClassifier 은 없는놈!!
OutputCodeClassifier 은 없는놈!!
PassiveAggressiveClassifier Acc:  [0.66666667 0.75       0.58333333 0.31428571 0.6       ] 평균 Acc: 0.5829
Perceptron Acc:  [0.61111111 0.80555556 0.47222222 0.48571429 0.62857143] 평균 Acc: 0.6006
QuadraticDiscriminantAnalysis Acc:  [0.97222222 1.         1.         1.         1.        ] 평균 Acc: 0.9944
RadiusNeighborsClassifier Acc:  [nan nan nan nan nan] 평균 Acc: nan
RandomForestClassifier Acc:  [1.         0.94444444 1.         0.97142857 1.        ] 평균 Acc: 0.9832
RidgeClassifier Acc:  [1.         1.         1.         0.97142857 1.        ] 평균 Acc: 0.9943
RidgeClassifierCV Acc:  [1.         1.         1.         0.97142857 1.        ] 평균 Acc: 0.9943
SGDClassifier Acc:  [0.38888889 0.69444444 0.63888889 0.62857143 0.62857143] 평균 Acc: 0.5959
SVC Acc:  [0.69444444 0.69444444 0.61111111 0.62857143 0.6       ] 평균 Acc: 0.6457
StackingClassifier 은 없는놈!!
VotingClassifier 은 없는놈!!
'''