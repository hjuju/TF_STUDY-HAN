import numpy as np
from icecream import ic
x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.array(range(1001, 1101))


ic(x1.shape, x2.shape, y1.shape) # ic| x1.shape: (100, 3), x2.shape: (100, 3), y1.shape: (100, 1)