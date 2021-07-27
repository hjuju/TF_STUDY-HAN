import pandas as pd
import numpy as np

# d = pd.read_csv('D:\STUDY/Dacon/_data/newstopic/predict210727_0206.csv')


x = np.load("./Dacon/_save/_npy/newstopic_x_data.npy",allow_pickle=True) 
y = np.load("./Dacon/_save/_npy/newstopic_y_data.npy")

print(y)