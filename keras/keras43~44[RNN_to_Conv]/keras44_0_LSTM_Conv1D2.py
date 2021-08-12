import numpy as np
from numpy.core.fromnumeric import transpose
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.datasets import load_boston
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score, mean_squared_error

x_data = np.array(range(1,101))
x_predict = np.array(range(96, 106))

size = 6
size1 = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(x_data, size)

x_predict = split_x(x_predict, size1) # (6, 5)

x = dataset[:, :-1] # (95, 5)  
y = dataset[:, -1] # (95,)
x_predict = x_predict[:, :-1]







# ic(x)
# ic(y)

ic(x.shape, y.shape)
# 시계열 데이터는 x와 y를 분리를 해줘야함


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=60) # train 309, test 133

# scaler = QuantileTransformer()
scaler = StandardScaler()
# scaler = PowerTransformer()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
ic(x_train.shape, x_test.shape)


# ic(np.unique(y))


model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(5,1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))