import numpy as np
from sklearn.datasets import load_boston
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore') # 경고무시

### 머신러닝(evaluate -> score)

datasets = load_boston()

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
ARDRegression 의 R2_score:  0.8012569266998071
AdaBoostRegressor 의 R2_score:  0.9040001729679632
BaggingRegressor 의 R2_score:  0.9257866286272785
BayesianRidge 의 R2_score:  0.7937918622384766
CCA 의 R2_score:  0.7913477184424628
DecisionTreeRegressor 의 R2_score:  0.7189125149575053
DummyRegressor 의 R2_score:  -0.0005370164400797517
ElasticNet 의 R2_score:  0.7338335519267194
ElasticNetCV 의 R2_score:  0.7167760356856181
ExtraTreeRegressor 의 R2_score:  0.8034147040637956
ExtraTreesRegressor 의 R2_score:  0.9342379869486193
GammaRegressor 의 R2_score:  -0.0005370164400797517
GaussianProcessRegressor 의 R2_score:  -6.073105259620457
GradientBoostingRegressor 의 R2_score:  0.9448291375313237
HistGradientBoostingRegressor 의 R2_score:  0.9323597806119726
HuberRegressor 의 R2_score:  0.7422362923023305
IsotonicRegression 은 없는놈!!
KNeighborsRegressor 의 R2_score:  0.5900872726222293
KernelRidge 의 R2_score:  0.8333325493741729
Lars 의 R2_score:  0.7746736096721596
LarsCV 의 R2_score:  0.7981576314184017
Lasso 의 R2_score:  0.7240751024070102
LassoCV 의 R2_score:  0.7517507753137198
LassoLars 의 R2_score:  -0.0005370164400797517
LassoLarsCV 의 R2_score:  0.8127604328474287
LassoLarsIC 의 R2_score:  0.8131423868817642
LinearRegression 의 R2_score:  0.8111288663608656
LinearSVR 의 R2_score:  0.30451912898890676
MLPRegressor 의 R2_score:  0.64913622809098
MultiOutputRegressor 은 없는놈!!
MultiTaskElasticNet 은 없는놈!!
MultiTaskElasticNetCV 은 없는놈!!
MultiTaskLasso 은 없는놈!!
MultiTaskLassoCV 은 없는놈!!
NuSVR 의 R2_score:  0.2594558622083819
OrthogonalMatchingPursuit 의 R2_score:  0.5827617571381449
OrthogonalMatchingPursuitCV 의 R2_score:  0.78617447738729
PLSCanonical 의 R2_score:  -2.2317079741425756
PLSRegression 의 R2_score:  0.8027313142007887
PassiveAggressiveRegressor 의 R2_score:  -0.23433078084748438
PoissonRegressor 의 R2_score:  0.8575511295761173
RANSACRegressor 의 R2_score:  0.6650755606173373
RadiusNeighborsRegressor 은 없는놈!!
RandomForestRegressor 의 R2_score:  0.9224839377114601
RegressorChain 은 없는놈!!
Ridge 의 R2_score:  0.8098487632912242
RidgeCV 의 R2_score:  0.8112529186350843
SGDRegressor 의 R2_score:  -6.055376884962699e+26
SVR 의 R2_score:  0.23474677555722312
StackingRegressor 은 없는놈!!
TheilSenRegressor 의 R2_score:  0.7940542412257224
TransformedTargetRegressor 의 R2_score:  0.8111288663608656
TweedieRegressor 의 R2_score:  0.7416351559639474
VotingRegressor 은 없는놈!!
'''
