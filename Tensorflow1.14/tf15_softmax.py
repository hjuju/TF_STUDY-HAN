from re import X
import numpy as np
import tensorflow as tf
from tensorflow.python.client.session import Session
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.python.training import optimizer
tf.compat.v1.set_random_seed(66)


x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]  # (8,4)

y_data = [[0, 0, 1],   # 2
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],   # 1
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],   # 0
          [1, 0, 0]] # 원 핫 인코딩 되어있는 상태 (8,3)

x = tf.placeholder(tf.float32, shape=(None, 4))
y = tf.placeholder(tf.float32, shape=(None, 3))

w = tf.Variable(tf.random.normal([4, 3], name='weight')) 
b = tf.Variable(tf.random.normal([1, 3], name='bias')) # 머릿속으로 w에대한 첫번째 두번째 레이어 그려보기 

hypothesis = tf.nn.softmax(tf.matmul(x, w)+b)

# categorical_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))


# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
# train = optimizer.minimize(loss)
# optimizer + train 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)


sess = tf.Session()
sess.run(global_variables_initializer())

for steps in range(2001):
    _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_data, y:y_data})
    if steps % 200 == 0:
        print(steps, cost_val)

# predict
results = sess.run(hypothesis, feed_dict={x:[[1, 11, 7, 9]]})
print(results, sess.run(tf.argmax(results, 1)))

sess.close()