import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 - 4 * x + 6
x = np.linspace(-1, 6, 100) # -1~ 6까지 총 100개의 데이터
print(x)

y = f(x) # x에 매치된 100개의 y값 생성

plt.plot(x,y, 'k-')
plt.plot(2,2, 'sk')
plt.grid()
plt.xlabel('x')
plt.xlabel('y')
plt.show()