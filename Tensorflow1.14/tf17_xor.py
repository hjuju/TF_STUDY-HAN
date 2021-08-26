import tensorflow as tf
tf.set_random_seed(66)

x_data = [[0,0], [0,1], [1,0], [1,1]] # (4, 2)
y_data = [[0], [1], [1], [0]]         # (4, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) # shape 맞춰줌 (행무시 열우선)

w = tf.Variable(tf.random.normal([2, 1], name='weight')) # (2,1)
b = tf.Variable(tf.random.normal([1], name='bias'))

# 행렬 연산 -> 앞의 열과 뒤의 행의 숫자가 같으면 가능 ex(2,2) * (2,1) -> (2,1)

hypothesis = tf.sigmoid(tf.matmul(x, w) + b) # 출력되는 y값에 tf.sigmoid로 묶어서 0과1사이로 출력

cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1- y)*tf.log(1-hypothesis)) # binary_crossentropy 결과값이 -로 나와서 -붙여줘야함

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) 
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

#3 훈련
for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                                    feed_dict={x:x_data,  y:y_data})
    if epochs % 100 == 0:
        print(epochs, 'cost: ', cost_val, '\n', hy_val) # nan이 나오는이유 => 러닝레이트가 너무 커서

#4 평가, 예측
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) # 조건에 부합하면 1, 조건에 부합하지않으면 0
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

c, a = sess.run([predicted, accuracy], feed_dict={x:x_data, y:y_data})
print('==============================================================')
print("예측값: \n", hy_val, "\n 원래값: \n", c, "\n Acc: ", a)
sess.close()