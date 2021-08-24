# 실습 
# 1. [4]
# 2. [5, 6]
# 3. [6, 7, 8]

import tensorflow as tf
tf.set_random_seed(66)

# x_train = [1,2,3]
# y_train = [1,2,3]

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32) # 랜덤하게 내맘대로 넣어준 값
b = tf.Variable(tf.compat.v1.random_normal([1]), dtype= tf.float32) # 초기값

hypothesis = x_train * W + b # 모델구현
# f(x) = wx + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse
            # 평균        제곱      오차   => mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss) # loss의 최소값을 찾아줌

#### 그래프 형태로 만들어줌 출력하려면 sessrun에 넣어줘야함
# sess = tf.Session()

sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 출력하기전에 gvi 해줌

for step in range(2001): # -> 2000번 epochs
    # print(step, sess.run(loss), sess.run(W), sess.run(b))
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b],
                                          feed_dict={x_train:[1,2,3], y_train:[3,5,7]}) # 위에서 한번에 엮어서 sess.run
    if step % 20 == 0: # 20번마다 출력 시킴
        print(step, loss_val, W_val, b_val)

# predict 하는 코드 추가
# x_tes라는 placeholder 생성

x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

hypothesis2 = x_test * W_val + b_val
pred1 = sess.run([hypothesis2], feed_dict={x_test:[4]})
pred2 = sess.run([hypothesis2], feed_dict={x_test:[5,6]})
pred3 = sess.run([hypothesis2], feed_dict={x_test:[6,7,8]})
print(pred1)
print(pred2)
print(pred3)

# [array([8.860945], dtype=float32)]
# [array([10.780415, 12.699885], dtype=float32)]
# [array([12.699885, 14.619356, 16.538828], dtype=float32)]

# 실습 
# 1. [4]
# 2. [5, 6]
# 3. [6, 7, 8]


