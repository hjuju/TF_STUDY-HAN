# coefficient 계수

x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-3, 65, -19, 11, 3, 47, -1, -7, -47, -25]

# import matplotlib.pyplot as plt
# plt.plot(x,y)
# plt.show()

import pandas as pd
from icecream import ic
df = pd.DataFrame({'X':x,'Y':y})
# ic(df)
# ic(df.shape)

x_train = df.loc[:,'X']
y_train = df.loc[:,'Y']
# ic(x_train.shape, y_train.shape) # (10, ), (10, )
x_train = x_train.values.reshape(len(x_train),1) # (10, ) -> (10,1) 넘파이로 바꾼 뒤 리쉐잎
ic(x_train.shape, y_train.shape) #  x_train.shape: (10, 1), y_train.shape: (10,)


from sklearn.linear_model import LinearRegression
#2. 모델
model = LinearRegression()
model.fit(x_train,y_train)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_train, y_train)
ic(score)

print("기울기: ", model.coef_)
print("절편: ", model.intercept_)

# 기울기:  [2.]
# 절편:  3.0