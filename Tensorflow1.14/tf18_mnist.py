from keras import datasets
from keras.datasets import mnist
import tensorflow as tf

(x_train, x_test), (y_train, y_test) = mnist.load_data

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None, 28 * 28])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None, 10]) # shape 맞춰줌 (행무시 열우선)

# 히든레이어1
w1 = tf.Variable(tf.random.normal([28 * 28, 1], name='weight')) # 2 * 3 = 6 (레이어의 개수)
b1 = tf.Variable(tf.random.normal([10], name='bias'))

layer1 = tf.sigmoid(tf.matmul(x_train, w1) + b1) # activation

# 아웃풋레이어

w2 = tf.Variable(tf.random.normal([8, 1], name='weight')) # (2,1)
b2 = tf.Variable(tf.random.normal([1], name='bias'))

hypothesis = tf.sigmoid(tf.matmul(layer1, w2) + b2)

cost = -tf.reduce_mean(y_train*tf.log(hypothesis)+(1- y_train)*tf.log(1-hypothesis)) # binary_crossentropy 결과값이 -로 나와서 -붙여줘야함

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)