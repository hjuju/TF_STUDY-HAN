import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from icecream import ic

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

# model.summary()

# ic(model.weights)
'''
 [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[-1.0118068 , -0.48295105, -0.68460524]], dtype=float32)>, kernel == weight
                    <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.] bias -> 3 , dtype=float32)>,
                    <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
                   array([[ 0.8416804 , -0.6662749 ],
                          [ 0.8009095 ,  0.45825732],
                          [ 0.8217516 , -0.70994925]], dtype=float32)>,
                    <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>,
                    <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
                   array([[1.0874716],
                          [1.1409198]], dtype=float32)>,
                    <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

'''
print("="*100)
# ic(model.trainable_weights)

ic(len(model.weights))
ic(len(model.trainable_weights)) #6 -> 층마다 w, b 하나씩
