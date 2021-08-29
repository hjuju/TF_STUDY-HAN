from keras import datasets
from keras.datasets import mnist
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28 * 28).astype('float32')/255 # 3차원 -> 4차원  // 데이터의 내용과 순서가 바뀌면 안됨
x_test = x_test.reshape(10000, 28 * 28).astype('float32')/255
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()


x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28 * 28])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10]) # shape 맞춰줌 (행무시 열우선)

# 히든레이어1
w1 = tf.Variable(tf.random.normal([28 * 28, 512], name='weight'))
b1 = tf.Variable(tf.random.normal([512], name='bias'))

layer1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1) # activation

w2 = tf.Variable(tf.random.normal([512, 256], name='weight')) 
b2 = tf.Variable(tf.random.normal([256], name='bias'))

layer2 = tf.nn.sigmoid(tf.matmul(layer1, w2) + b2) # activation

w3 = tf.Variable(tf.random.normal([256, 128], name='weight')) 
b3 = tf.Variable(tf.random.normal([128], name='bias'))

layer3 = tf.nn.sigmoid(tf.matmul(layer2, w3) + b3) # activation

w4 = tf.Variable(tf.random.normal([128, 64], name='weight')) 
b4 = tf.Variable(tf.random.normal([64], name='bias'))

layer4 = tf.nn.sigmoid(tf.matmul(layer3, w4) + b4) # activation

# 아웃풋레이어

w5 = tf.Variable(tf.random.normal([64, 10], name='weight')) 
b5 = tf.Variable(tf.random.normal([10], name='bias'))

output = tf.nn.softmax(tf.matmul(layer4, w5) + b5)

# hypothesis = tf.nn.elu(tf.matmul(layer2, w3) + b3)
# hypothesis = tf.nn.selu(tf.matmul(layer2, w3) + b3)
# hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3)
# hypothesis = tf.nn.dropout(받게되는 layer

cost = tf.losses.softmax_cross_entropy(y, output) # categorical_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

#3 훈련

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(125):
    cost_val, hy_val, _ = sess.run([cost, output, train], 
              feed_dict={x:x_train, y:y_train})
    if epochs % 10 == 0:
        print(epochs, ", cost : ", cost_val, "\n", hy_val)

# 4. 평가, 예측

predicted = tf.cast(output > 0.5, dtype=tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

c, a = sess.run([predicted, accuracy], feed_dict={x:x_train, y:y_train})

print("예측 값 : \n", hy_val,
     '\n 예측 결과값 : \n', c, "\n Accuracy : ", a)
sess.close()

#