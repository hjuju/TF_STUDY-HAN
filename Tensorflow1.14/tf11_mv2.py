import tensorflow as tf
tf.compat.v1.set_random_seed(66)

x_data = [[73, 51, 65],
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]
# (5, 3)

y_data = [[152], [185], [180], [205],[142]] # (5, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) # shape 맞춰줌 (행무시 열우선)

w = tf.Variable(tf.random.normal([3, 1], name='weight')) # (3,1)
b = tf.Variable(tf.random.normal([1], name='bias'))

# 행렬 연산 -> 앞의 열과 뒤의 행의 숫자가 같으면 가능 ex(2,2) * (2,1) -> (2,1)

hypothesis = tf.matmul(x, w) + b # matmul -> 행렬연산 할 때 사용

cost = tf.reduce_mean(tf.square(hypothesis - y)) # mse

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5) # 1e-5 = 0.00001 1의 -5승

train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
                                    feed_dict={x:x_data,  y:y_data})
    if epochs % 10 == 0:
        print(epochs, 'cost: ', cost_val, '\n', hy_val) # nan이 나오는이유 => 러닝레이트가 너무 커서

sess.close()