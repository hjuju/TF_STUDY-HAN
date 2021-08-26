from re import X
import numpy as np
from sklearn import datasets
import tensorflow as tf
from tensorflow.python.client.session import Session
from tensorflow.python.ops.variables import global_variables_initializer
from tensorflow.python.training import optimizer
tf.compat.v1.set_random_seed(66)
from sklearn.datasets import load_iris, load_wine
from icecream import ic
from sklearn.model_selection import train_test_split

datasets = load_wine()

x_data = datasets.data
y_data = datasets.target

ic(x_data.shape, y_data.shape) # ic| x_data.shape: (178, 13), y_data.shape: (178,)

x = tf.placeholder(tf.float32, shape=(None, 13))
y = tf.placeholder(tf.float32, shape=(None, ))

w = tf.Variable(tf.random.normal([13, 1], name='weight')) 
b = tf.Variable(tf.random.normal([1], name='bias')) # 머릿속으로 w에대한 첫번째 두번째 레이어 그려보기 

# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=66, shuffle=True)

hypothesis = tf.nn.softmax(tf.matmul(x, w) +b)

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
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

c, a = sess.run([predicted, accuracy], feed_dict={x:x_data, y:y_data})
print("Accuracy : ", a)
sess.close()
