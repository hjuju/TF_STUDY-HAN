import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

x = np.arange(1,5)
y = softmax(x)

plt.pie(y, labels=y, shadow=True, startangle=90)
plt.show() # 전체 데이터 중에 각각의 값을 총합을 1로하여 분류