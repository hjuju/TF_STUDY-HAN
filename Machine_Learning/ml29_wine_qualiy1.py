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

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBClassifier(n_jobs=-1)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)

print('acc: ', score) 

# acc:  0.6816326530612244