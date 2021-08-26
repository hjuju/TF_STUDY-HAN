# perceotron -> mlp

import tensorflow as tf
from tensorflow.python import training
tf.set_random_seed(66)

x_data = [[0,0], [0,1], [1,0], [1,1]] # (4, 2)
y_data = [[0], [1], [1], [0]]         # (4, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) # shape 맞춰줌 (행무시 열우선)

# 히든레이어1
w1 = tf.Variable(tf.random.normal([2, 12], name='weight')) # 2 * 3 = 6 (레이어의 개수)
b1 = tf.Variable(tf.random.normal([12], name='bias'))

layer1 = tf.sigmoid(tf.matmul(x, w1) + b1) # activation

# 히든레이어2
w2 = tf.Variable(tf.random.normal([12, 10], name='weight')) # 2 * 3 = 6 (레이어의 개수)
b2 = tf.Variable(tf.random.normal([10], name='bias'))

layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2) # activation

# 아웃풋레이어

w3 = tf.Variable(tf.random.normal([10, 1], name='weight')) # (2,1)
b3 = tf.Variable(tf.random.normal([1], name='bias'))

output = tf.sigmoid(tf.matmul(layer2, w3) + b3)

cost = -tf.reduce_mean(y*tf.log(output)+(1- y)*tf.log(1-output)) # binary_crossentropy 결과값이 -로 나와서 -붙여줘야함

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

#3 훈련
predicted = tf.cast(output > 0.5, dtype=tf.float32 )
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(6001):
    cost_val, hy_val, _ = sess.run([cost, output, train],
        feed_dict={x:x_data, y:y_data})
    if epochs % 1000 == 0:
        print(epochs, "cost :", cost_val, "\n", hy_val)

h, c, a = sess.run([output, predicted, accuracy], feed_dict = {x:x_data,y:y_data})
print("Hypothesis : \n", h, "\npredict : \n" ,c , "\n Accuarcy : ",a)

sess.close()