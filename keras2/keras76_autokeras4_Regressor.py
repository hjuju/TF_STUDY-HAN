from sklearn.datasets import load_boston
import autokeras as ak
import pandas as pd


# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target


# 2. 모델
model = ak.StructureDataRegressor(overwrite=True, max_trials=1)


# 3. 훈련
model.fit(x, y, epochs=2, validation_split=0.2)

# 4. 평가, 예측
results = model.evaluate(x, y)