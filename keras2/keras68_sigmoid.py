import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)

ic(len(x))

y = sigmoid(x)

plt.plot(x,y)
plt.grid()
plt.show()