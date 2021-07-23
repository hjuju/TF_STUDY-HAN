import pandas as pd
from icecream import ic
import numpy as np
from pandas.core.tools.datetimes import Scalar
from tensorflow.python.keras.backend import concatenate, reshape, transpose
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Input , Conv1D, Concatenate, Flatten, Dropout
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from icecream import ic
import time
import datetime


ss = pd.read_csv('./samsung/_data/SAMSUNG.csv', header=0,  nrows=2601, encoding='CP949')
sk = pd.read_csv('./samsung/_data/SK.csv', header=0,  nrows=2601, encoding='CP949')

ss = ss[['고가','저가','거래량','종가', '시가']]   
sk = sk[['고가','저가','거래량','종가', '시가']]

ic(ss, sk) # 고가 저가 거래량 종가 시가
# ic(ss.shape, sk.shape) # ss.shape: (2601, 5), sk.shape: (2601, 5)


# 오름차순으로 정렬, 배열로 변경
ss = ss.sort_index(ascending=False).to_numpy()
sk = sk.sort_index(ascending=False).to_numpy()

# ic(ss, sk)

size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

split_samsung = split_x(ss, size)
split_sk = split_x(sk, size)

# ic(split_samsung, split_sk)
# ic(split_samsung.shape, split_sk.shape) # split_samsung.shape: (2597, 5, 5), split_sk.shape: (2597, 5, 5)


# ic(x1_pred.shape, x2_pred.shape) # x1_pred.shape: (5, 5, 5), x2_pred.shape: (5, 5, 5)



