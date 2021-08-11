import numpy as np
from sklearn.datasets import load_diabetes
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore') # 경고무시
from sklearn.model_selection import KFold, cross_val_score

datasets = load_diabetes()


# 1. 데이터
x = datasets.data
y = datasets.target
# ic(x.shape, y.shape)  

# 2. 모델(머신러닝에서는 정의만 해주면 됨)

allAlgorithms = all_estimators(type_filter='regressor')
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
모델의 개수: 54
ARDRegression Acc:  [0.49874835 0.48765748 0.56284846 0.37728801 0.53474369] 평균 Acc: 0.4923
AdaBoostRegressor Acc:  [0.40707605 0.42577404 0.53643948 0.37765992 0.45231823] 평균 Acc: 0.4399
BaggingRegressor Acc:  [0.26576221 0.41786701 0.47190879 0.34246342 0.32223736] 평균 Acc: 0.364
BayesianRidge Acc:  [0.50082189 0.48431051 0.55459312 0.37600508 0.5307344 ] 평균 Acc: 0.4893
CCA Acc:  [0.48696409 0.42605855 0.55244322 0.21708682 0.50764701] 평균 Acc: 0.438
DecisionTreeRegressor Acc:  [-0.257969   -0.1486278  -0.13925131 -0.05109302 -0.02097735] 평균 Acc: -0.1236
DummyRegressor Acc:  [-1.54258856e-04 -2.98519672e-03 -1.53442062e-05 -3.80334913e-03
 -9.58335111e-03] 평균 Acc: -0.0033
ElasticNet Acc:  [ 0.00810127  0.00637294  0.00924848  0.0040621  -0.00081988] 평균 Acc: 0.0054
ElasticNetCV Acc:  [0.43071558 0.461506   0.49133954 0.35674829 0.4567084 ] 평균 Acc: 0.4394
ExtraTreeRegressor Acc:  [-0.12058477 -0.10470216 -0.11276873 -0.00017631 -0.04451664] 평균 Acc: -0.0765
ExtraTreesRegressor Acc:  [0.38719501 0.46755124 0.54073442 0.37730584 0.44974144] 평균 Acc: 0.4445
GammaRegressor Acc:  [ 0.00523561  0.00367973  0.0060814   0.00174734 -0.00306898] 평균 Acc: 0.0027
GaussianProcessRegressor Acc:  [ -5.6360757  -15.27401119  -9.94981439 -12.46884878 -12.04794389] 평균 Acc: -11.0753
GradientBoostingRegressor Acc:  [0.39145439 0.48618033 0.48128625 0.39707234 0.4466913 ] 평균 Acc: 0.4405
HistGradientBoostingRegressor Acc:  [0.28899498 0.43812684 0.51713242 0.37267554 0.35643755] 평균 Acc: 0.3947
HuberRegressor Acc:  [0.50334705 0.47508237 0.54650576 0.36883712 0.5173073 ] 평균 Acc: 0.4822
IsotonicRegression Acc:  [nan nan nan nan nan] 평균 Acc: nan
KNeighborsRegressor Acc:  [0.39683913 0.32569788 0.43311217 0.32635899 0.35466969] 평균 Acc: 0.3673
KernelRidge Acc:  [-3.38476443 -3.49366182 -4.0996205  -3.39039111 -3.60041537] 평균 Acc: -3.5938
Lars Acc:  [ 0.49198665 -0.66475442 -1.04410299 -0.04236657  0.51190679] 평균 Acc: -0.1495
LarsCV Acc:  [0.4931481  0.48774421 0.55427158 0.38001456 0.52413596] 평균 Acc: 0.4879
Lasso Acc:  [0.34315574 0.35348212 0.38594431 0.31614536 0.3604865 ] 평균 Acc: 0.3518
LassoCV Acc:  [0.49799859 0.48389346 0.55926851 0.37740074 0.51636393] 평균 Acc: 0.487
LassoLars Acc:  [0.36543887 0.37812653 0.40638095 0.33639271 0.38444891] 평균 Acc: 0.3742
LassoLarsCV Acc:  [0.49719648 0.48426377 0.55975856 0.37984022 0.51190679] 평균 Acc: 0.4866
LassoLarsIC Acc:  [0.49940515 0.49108789 0.56130589 0.37942384 0.5247894 ] 평균 Acc: 0.4912
LinearRegression Acc:  [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679] 평균 Acc: 0.4876
LinearSVR Acc:  [-0.33470258 -0.31629592 -0.41227819 -0.30276155 -0.47800225] 평균 Acc: -0.3688
MLPRegressor Acc:  [-2.76931352 -3.04065264 -3.32347082 -2.87111456 -3.28922056] 평균 Acc: -3.0588
MultiOutputRegressor 은 없는놈!!
MultiTaskElasticNet Acc:  [nan nan nan nan nan] 평균 Acc: nan
MultiTaskElasticNetCV Acc:  [nan nan nan nan nan] 평균 Acc: nan
MultiTaskLasso Acc:  [nan nan nan nan nan] 평균 Acc: nan
MultiTaskLassoCV Acc:  [nan nan nan nan nan] 평균 Acc: nan
NuSVR Acc:  [0.14471275 0.17351835 0.18539957 0.13894135 0.1663745 ] 평균 Acc: 0.1618
OrthogonalMatchingPursuit Acc:  [0.32934491 0.285747   0.38943221 0.19671679 0.35916077] 평균 Acc: 0.3121
OrthogonalMatchingPursuitCV Acc:  [0.47845357 0.48661326 0.55695148 0.37039612 0.53615516] 평균 Acc: 0.4857
PLSCanonical Acc:  [-0.97507923 -1.68534502 -0.8821301  -1.33987816 -1.16041996] 평균 Acc: -1.2086
PLSRegression Acc:  [0.47661395 0.4762657  0.5388494  0.38191443 0.54717873] 평균 Acc: 0.4842
PassiveAggressiveRegressor Acc:  [0.44486902 0.49243571 0.53221998 0.348829   0.48322501] 평균 Acc: 0.4603
PoissonRegressor Acc:  [0.32061441 0.35803358 0.3666005  0.28203414 0.34340626] 평균 Acc: 0.3341
RANSACRegressor Acc:  [0.17238385 0.13222315 0.21576084 0.04778566 0.089763  ] 평균 Acc: 0.1316
RadiusNeighborsRegressor Acc:  [-1.54258856e-04 -2.98519672e-03 -1.53442062e-05 -3.80334913e-03
 -9.58335111e-03] 평균 Acc: -0.0033
RandomForestRegressor Acc:  [0.36829919 0.48777631 0.46529284 0.37485675 0.41840163] 평균 Acc: 0.4229
RegressorChain 은 없는놈!!
Ridge Acc:  [0.40936669 0.44788406 0.47057299 0.34467674 0.43339091] 평균 Acc: 0.4212
RidgeCV Acc:  [0.49525464 0.48761091 0.55171354 0.3801769  0.52749194] 평균 Acc: 0.4884
SGDRegressor Acc:  [0.3933481  0.44180122 0.46452915 0.32961789 0.41529429] 평균 Acc: 0.4089
SVR Acc:  [0.14331635 0.18438697 0.17864042 0.1424597  0.1468719 ] 평균 Acc: 0.1591
StackingRegressor 은 없는놈!!
TheilSenRegressor Acc:  [0.50278188 0.46958722 0.55512433 0.35001211 0.53545186] 평균 Acc: 0.4826
TransformedTargetRegressor Acc:  [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679] 평균 Acc: 0.4876
TweedieRegressor Acc:  [ 0.00585525  0.00425899  0.00702558  0.00183408 -0.00315042] 평균 Acc: 0.0032
VotingRegressor 은 없는놈!!
'''