from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from icecream import ic

datasets = load_iris()

irisDF = pd.DataFrame(data= datasets.data, columns=datasets.feature_names)

# ic(irisDF)

Kmean = KMeans(n_clusters=3, max_iter=300, random_state=66) # 3개의 라벨을 추출(0,1,2)
Kmean.fit(irisDF)

results = Kmean.labels_ # Y값

# ic(results)

irisDF['cluster'] = Kmean.labels_ # 클러스터링 해서 생성한 y값
irisDF['target'] = datasets.target # 기존 y값

ic(datasets.feature_names)

# 생성한 y값과 기존y값 비교
for i in datasets.feature_names:
    iris_results = irisDF.groupby(['target', 'cluster'])[i].count()
    ic(i, iris_results)