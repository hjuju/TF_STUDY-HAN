import numpy as np
import pandas as pd
from scipy.sparse import data
from sklearn import datasets
from sklearn.datasets import load_wine
from icecream import ic
from xgboost import XGBClassifier

#1. 데이터
datasets = pd.read_csv('../_data/wine/winequality-white.csv', index_col=None, header=0, sep=';')

# ic(datasets.head())

# ic(datasets.shape) # (4898, 12)
# ic(datasets.describe())

datasets = datasets.values
# ic(datasets)

x = datasets[:,:11]
y = datasets[:,11]

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

def outliers(data_out):
    for i in data_out:
        quantile_1, q2, quantile_3 = np.percentile(data_out, [25,50,75])
        print("1사분위: ", quantile_1)
        print("q2: ", q2)
        print("3사분위: ", quantile_3)
        iqr = quantile_3 - quantile_1
        lower_bound = quantile_1 - (iqr *1.5)
        upper_bound = quantile_3 + (iqr *1.5)
        return np.where((data_out>upper_bound) | (data_out<lower_bound))

outliers_loc = outliers(x_train)

ic(outliers_loc)



# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
# model = XGBClassifier(n_jobs=-1)

# model.fit(x_train, y_train)

# score = model.score(x_test, y_test)

# print('acc: ', score) 

# # acc:  0.6816326530612244