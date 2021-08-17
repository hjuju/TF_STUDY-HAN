# [1, np.nan, np.nan, 8, 10]

'''
결측치 처리
1. 행삭제 -> 데이터가 적을때 데이터 손실이 큼 (엄청 많을때는 크게 상관없음)
2. nan대신 0넣기(특정값) -> [1, 0, 0, 8, 10]
                 앞에값 -> [1, 1, 1, 8, 10]
                 뒤에값 -> [1, 8, 8, 8, 10]
                 중위값 -> [1, 4.5, 4.5, 8, 10]
                 보간 ->   
                 보간: 리니어 이외의 결측값 데이터에 사용/ 결측치를 제거한 X데이터로 신뢰할 수 있는 모델을 돌려서 훈련시킴 -> 도출된 결과치에 predict(결측치)를 하여 결측치 값 도출
                 부스트계열은 결측치에 대해 자유롭다(ex. tree계열, )
3. 

'''

from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd
from icecream import ic

datastrs = ["8/13/2021","8/14/2021", "8/15/2021", "8/16/2021","8/17/2021"]
dates = pd.to_datetime(datastrs)
ic(dates)
ic(type(dates))
print("=*50")

ts = Series([1, np.nan, np.nan, 8, 10], index=dates)
ic(ts)

ts_intp_linear = ts.interpolate() # 결측치들이 중앙값으로 바꿔짐
ic(ts_intp_linear)