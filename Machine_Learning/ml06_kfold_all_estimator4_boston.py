import numpy as np
from sklearn.datasets import load_boston
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore') # 경고무시
from sklearn.model_selection import KFold, cross_val_score



datasets = load_boston()
print(datasets.DESCR)
print(datasets.feature_names)

# 1. 데이터
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)  

# 2. 모델(머신러닝에서는 정의만 해주면 됨)

allAlgorithms = all_estimators(type_filter='regressor')
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
ARDRegression Acc:  [0.80125693 0.76317071 0.56809285 0.6400258  0.71991866] 평균 Acc: 0.6985
AdaBoostRegressor Acc:  [0.89753132 0.8161707  0.80004042 0.84720174 0.89374763] 평균 Acc: 0.8509
BaggingRegressor Acc:  [0.89127481 0.83892351 0.81893038 0.84698922 0.89870112] 평균 Acc: 0.859
BayesianRidge Acc:  [0.79379186 0.81123808 0.57943979 0.62721388 0.70719051] 평균 Acc: 0.7038
CCA Acc:  [0.79134772 0.73828469 0.39419624 0.5795108  0.73224276] 평균 Acc: 0.6471
DecisionTreeRegressor Acc:  [0.80405748 0.68444815 0.79552832 0.73075023 0.81675233] 평균 Acc: 0.7663
DummyRegressor Acc:  [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911] 평균 Acc: -0.0135
ElasticNet Acc:  [0.73383355 0.76745241 0.59979782 0.60616114 0.64658354] 평균 Acc: 0.6708
ElasticNetCV Acc:  [0.71677604 0.75276545 0.59116613 0.59289916 0.62888608] 평균 Acc: 0.6565
ExtraTreeRegressor Acc:  [0.74324434 0.66763332 0.65940013 0.68081439 0.76913938] 평균 Acc: 0.704
ExtraTreesRegressor Acc:  [0.9353456  0.85419481 0.78033922 0.88493471 0.93450713] 평균 Acc: 0.8779
GammaRegressor Acc:  [-0.00058757 -0.03146716 -0.00463664 -0.02807276 -0.00298635] 평균 Acc: -0.0136
GaussianProcessRegressor Acc:  [-6.07310526 -5.51957093 -6.33482574 -6.36383476 -5.35160828] 평균 Acc: -5.9286
GradientBoostingRegressor Acc:  [0.9456135  0.83346581 0.82747163 0.88475476 0.93069849] 평균 Acc: 0.8844
HistGradientBoostingRegressor Acc:  [0.93235978 0.82415907 0.78740524 0.88879806 0.85766226] 평균 Acc: 0.8581
HuberRegressor Acc:  [0.74400323 0.64244715 0.52848946 0.37100122 0.63403398] 평균 Acc: 0.584
IsotonicRegression Acc:  [nan nan nan nan nan] 평균 Acc: nan
KNeighborsRegressor Acc:  [0.59008727 0.68112533 0.55680192 0.4032667  0.41180856] 평균 Acc: 0.5286
KernelRidge Acc:  [0.83333255 0.76712443 0.5304997  0.5836223  0.71226555] 평균 Acc: 0.6854
Lars Acc:  [0.77467361 0.79839316 0.5903683  0.64083802 0.68439384] 평균 Acc: 0.6977
LarsCV Acc:  [0.80141197 0.77573678 0.57807429 0.60068407 0.70833854] 평균 Acc: 0.6928
Lasso Acc:  [0.7240751  0.76027388 0.60141929 0.60458689 0.63793473] 평균 Acc: 0.6657
LassoCV Acc:  [0.71314939 0.79141061 0.60734295 0.61617714 0.66137127] 평균 Acc: 0.6779
LassoLars Acc:  [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911] 평균 Acc: -0.0135
LassoLarsCV Acc:  [0.80301044 0.77573678 0.57807429 0.60068407 0.72486787] 평균 Acc: 0.6965
LassoLarsIC Acc:  [0.81314239 0.79765276 0.59012698 0.63974189 0.72415009] 평균 Acc: 0.713
LinearRegression Acc:  [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 평균 Acc: 0.7128
LinearSVR Acc:  [0.72801523 0.77047227 0.43638225 0.4650722  0.62923739] 평균 Acc: 0.6058
MLPRegressor Acc:  [0.55112167 0.57313087 0.4518494  0.44432373 0.561887  ] 평균 Acc: 0.5165
MultiOutputRegressor 은 없는놈!!
MultiTaskElasticNet Acc:  [nan nan nan nan nan] 평균 Acc: nan
MultiTaskElasticNetCV Acc:  [nan nan nan nan nan] 평균 Acc: nan
MultiTaskLasso Acc:  [nan nan nan nan nan] 평균 Acc: nan
MultiTaskLassoCV Acc:  [nan nan nan nan nan] 평균 Acc: nan
NuSVR Acc:  [0.2594254  0.33427351 0.263857   0.11914968 0.170599  ] 평균 Acc: 0.2295
OrthogonalMatchingPursuit Acc:  [0.58276176 0.565867   0.48689774 0.51545117 0.52049576] 평균 Acc: 0.5343
OrthogonalMatchingPursuitCV Acc:  [0.75264599 0.75091171 0.52333619 0.59442374 0.66783377] 평균 Acc: 0.6578
PLSCanonical Acc:  [-2.23170797 -2.33245351 -2.89155602 -2.14746527 -1.44488868] 평균 Acc: -2.2096
PLSRegression Acc:  [0.80273131 0.76619347 0.52249555 0.59721829 0.73503313] 평균 Acc: 0.6847
PassiveAggressiveRegressor Acc:  [ 0.29969024  0.18920713 -0.14318513  0.19035349  0.1006669 ] 평균 Acc: 0.1273
PoissonRegressor Acc:  [0.85659255 0.8189989  0.66691488 0.67998192 0.75195656] 평균 Acc: 0.7549
RANSACRegressor Acc:  [0.60896865 0.72356776 0.44864916 0.02176365 0.60260022] 평균 Acc: 0.4811
RadiusNeighborsRegressor Acc:  [nan nan nan nan nan] 평균 Acc: nan
RandomForestRegressor Acc:  [0.91876886 0.84118362 0.81706348 0.88540717 0.90575334] 평균 Acc: 0.8736
RegressorChain 은 없는놈!!
Ridge Acc:  [0.80984876 0.80618063 0.58111378 0.63459427 0.72264776] 평균 Acc: 0.7109
RidgeCV Acc:  [0.81125292 0.80010535 0.58888304 0.64008984 0.72362912] 평균 Acc: 0.7128
SGDRegressor Acc:  [-2.88699789e+26 -9.81779129e+25 -3.10635607e+24 -8.45271938e+25
 -5.98788898e+25] 평균 Acc: -1.0687802834174426e+26
SVR Acc:  [0.23475113 0.31583258 0.24121157 0.04946335 0.14020554] 평균 Acc: 0.1963
StackingRegressor 은 없는놈!!
TheilSenRegressor Acc:  [0.78469087 0.71668838 0.58717787 0.5486828  0.71878782] 평균 Acc: 0.6712
TransformedTargetRegressor Acc:  [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 평균 Acc: 0.7128
TweedieRegressor Acc:  [0.7492543  0.75457294 0.56286929 0.57989884 0.63242475] 평균 Acc: 0.6558
VotingRegressor 은 없는놈!!
'''