from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from icecream import ic

dataset = load_boston()

x = dataset.data # 506개의 데이터(집값)에 대한 13개의 특성 (506, 13)
y = dataset.target # 집 값에 대한 데이터 506개 (506,)

ic(x.shape)
ic(y.shape)

ic(dataset.feature_names)
ic(dataset.data)
ic(dataset.DESCR) # 데이터셋 기술서 

model = Sequential()


