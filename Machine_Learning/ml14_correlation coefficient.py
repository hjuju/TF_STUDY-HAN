import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
from icecream import ic

datasets = load_iris()

# ic(datasets.keys())
# ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename']

# ic(datasets.target_names)
# ['setosa', 'versicolor', 'virginica']

x = datasets.data
y = datasets.target

ic(x.shape, y.shape) # x.shape: (150, 4), y.shape: (150,)

df = pd.DataFrame(x, columns=datasets['feature_names'])
# ic(df)

# y컬럼 추가 
df['Target'] = y
ic(df.head())

print("============================= 상관계수 히트 맵 =============================")
ic(df.corr()) # 데이터끼리의 상관관계 

'''
ic| df.corr():                    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)    Target
               sepal length (cm)           1.000000         -0.117570           0.871754          0.817941  0.782561
               sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126 -0.426658
               petal length (cm)           0.871754         -0.428440           1.000000          0.962865  0.949035
               petal width (cm)            0.817941         -0.366126           0.962865          1.000000  0.956547
               Target                      0.782561         -0.426658           0.949035          0.956547  1.000000

petal length 와 petal width는 상관관계가 높다. 이 표를 보고 상관관계가 적은건 컬럼을 지우는 조치 필요     

'''

import matplotlib.pyplot as plt
import seaborn as sns # 알아보기

# 상관관계 시각화(영향을 미치는 컬럼과 영향이 없는 컬럼을 구분함)
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()