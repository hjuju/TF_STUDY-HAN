import numpy as np
from sklearn.datasets import load_diabetes
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore') # 경고무시

### 머신러닝(evaluate -> score)

datasets = load_diabetes()

# 1. 데이터
x = datasets.data
y = datasets.target
ic(x.shape, y.shape)  # (150, 4), (150,)->(150, 3)
ic(y.shape)   # (150, )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# 1-2. 데이터 전처리
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델(머신러닝에서는 정의만 해주면 됨)

allAlgorithms = all_estimators(type_filter='regressor')
# ic(allAlgorithms)
print('모델의 개수:',len(allAlgorithms))

for (name , algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train) 

        y_predict = model.predict(x_test) 
        # acc = accuracy_score(y_test,y_predict)
        r2 = r2_score(y_test, y_predict)
        print(name,'의 R2_score: ', r2)
    except:
        # continue
        print(name,'은 없는놈!!')
# predict는 100퍼센트 다 있음 가끔 score가 없는 경우도 있음
# try, except로 에러뜬거 무시하고 계속해서 정상적으로 출력

'''
모델의 개수: 54
ARDRegression 의 R2_score:  0.49874835036886966
AdaBoostRegressor 의 R2_score:  0.38216798576432354
BaggingRegressor 의 R2_score:  0.39142471241125576
BayesianRidge 의 R2_score:  0.5008218932350128
CCA 의 R2_score:  0.48696409064967594
DecisionTreeRegressor 의 R2_score:  -0.24832761141784276
DummyRegressor 의 R2_score:  -0.00015425885559339214
ElasticNet 의 R2_score:  0.008101269711286885
ElasticNetCV 의 R2_score:  0.43071557917754755
ExtraTreeRegressor 의 R2_score:  0.030511046905312145
ExtraTreesRegressor 의 R2_score:  0.3980487043494143
GammaRegressor 의 R2_score:  0.005812599388535289
GaussianProcessRegressor 의 R2_score:  -5.636096404472059
GradientBoostingRegressor 의 R2_score:  0.38961195049592856
HistGradientBoostingRegressor 의 R2_score:  0.28899497703380905
HuberRegressor 의 R2_score:  0.5033469327985092
IsotonicRegression 은 없는놈!!
KNeighborsRegressor 의 R2_score:  0.3968391279034368
KernelRidge 의 R2_score:  -3.3847644323549924
Lars 의 R2_score:  0.4919866521464168
LarsCV 의 R2_score:  0.5010892359535759
Lasso 의 R2_score:  0.3431557382027084
LassoCV 의 R2_score:  0.49757816595208426
LassoLars 의 R2_score:  0.36543887418957965
LassoLarsCV 의 R2_score:  0.49519427906782676
LassoLarsIC 의 R2_score:  0.4994051517531072
LinearRegression 의 R2_score:  0.5063891053505035
LinearSVR 의 R2_score:  -0.33470258280275056
MLPRegressor 의 R2_score:  -2.871945947667631
MultiOutputRegressor 은 없는놈!!
MultiTaskElasticNet 은 없는놈!!
MultiTaskElasticNetCV 은 없는놈!!
MultiTaskLasso 은 없는놈!!
MultiTaskLassoCV 은 없는놈!!
NuSVR 의 R2_score:  0.14471275169122289
OrthogonalMatchingPursuit 의 R2_score:  0.3293449115305741
OrthogonalMatchingPursuitCV 의 R2_score:  0.44354253337919747
PLSCanonical 의 R2_score:  -0.9750792277922911
PLSRegression 의 R2_score:  0.4766139460349792
PassiveAggressiveRegressor 의 R2_score:  0.45611736428368466
PoissonRegressor 의 R2_score:  0.32989738735862917
RANSACRegressor 의 R2_score:  0.29132946288528216
RadiusNeighborsRegressor 의 R2_score:  -0.00015425885559339214
RandomForestRegressor 의 R2_score:  0.3695483594335546
RegressorChain 은 없는놈!!
Ridge 의 R2_score:  0.40936668956159705
RidgeCV 의 R2_score:  0.49525463889305044
SGDRegressor 의 R2_score:  0.39326997297232424
SVR 의 R2_score:  0.14331518075345895
StackingRegressor 은 없는놈!!
TheilSenRegressor 의 R2_score:  0.5018754182042446
TransformedTargetRegressor 의 R2_score:  0.5063891053505035
TweedieRegressor 의 R2_score:  0.005855247171688949
VotingRegressor 은 없는놈!!

Standarf Scarler
ARDRegression 의 R2_score:  0.4987481092636239
AdaBoostRegressor 의 R2_score:  0.3964852239188823
BaggingRegressor 의 R2_score:  0.3271421188419379
BayesianRidge 의 R2_score:  0.5007397987045734
CCA 의 R2_score:  0.4869640906496754
DecisionTreeRegressor 의 R2_score:  -0.21797340899528317
DummyRegressor 의 R2_score:  -0.00015425885559339214
ElasticNet 의 R2_score:  0.460731586257346
ElasticNetCV 의 R2_score:  0.4992225821392484
ExtraTreeRegressor 의 R2_score:  -0.4378556987963458
ExtraTreesRegressor 의 R2_score:  0.3818027938081541
GammaRegressor 의 R2_score:  0.4279999818217677
GaussianProcessRegressor 의 R2_score:  -0.8233124107173797
GradientBoostingRegressor 의 R2_score:  0.38746323660301707
HistGradientBoostingRegressor 의 R2_score:  0.28899497703380905
HuberRegressor 의 R2_score:  0.5070222171288641
IsotonicRegression 은 없는놈!!
KNeighborsRegressor 의 R2_score:  0.38626977834604637
KernelRidge 의 R2_score:  -3.2335362088031383
Lars 의 R2_score:  0.49198665214641657
LarsCV 의 R2_score:  0.5010892359535756
Lasso 의 R2_score:  0.49728444294444063
LassoCV 의 R2_score:  0.49751056665793925
LassoLars 의 R2_score:  0.36543887418957954
LassoLarsCV 의 R2_score:  0.4951942790678253
LassoLarsIC 의 R2_score:  0.49940515175310707
LinearRegression 의 R2_score:  0.5063891053505039
LinearSVR 의 R2_score:  0.350299309283729
MLPRegressor 의 R2_score:  -1.0049581535422139
MultiOutputRegressor 은 없는놈!!
MultiTaskElasticNet 은 없는놈!!
MultiTaskElasticNetCV 은 없는놈!!
MultiTaskLasso 은 없는놈!!
MultiTaskLassoCV 은 없는놈!!
NuSVR 의 R2_score:  0.1450699779346516
OrthogonalMatchingPursuit 의 R2_score:  0.3293449115305742
OrthogonalMatchingPursuitCV 의 R2_score:  0.44354253337919747
PLSCanonical 의 R2_score:  -0.9750792277922906
PLSRegression 의 R2_score:  0.4766139460349792
PassiveAggressiveRegressor 의 R2_score:  0.47267178008191535
PoissonRegressor 의 R2_score:  0.49583665918803266
RANSACRegressor 의 R2_score:  0.36305176142252926
RadiusNeighborsRegressor 은 없는놈!!
RandomForestRegressor 의 R2_score:  0.35810314285375255
RegressorChain 은 없는놈!!
Ridge 의 R2_score:  0.5056192903386312
RidgeCV 의 R2_score:  0.5019251143259338
SGDRegressor 의 R2_score:  0.5015895701214773
SVR 의 R2_score:  0.14351825322507306
StackingRegressor 은 없는놈!!
TheilSenRegressor 의 R2_score:  0.5031747964894072
TransformedTargetRegressor 의 R2_score:  0.5063891053505039
TweedieRegressor 의 R2_score:  0.42684234728812454
VotingRegressor 은 없는놈!!
'''
