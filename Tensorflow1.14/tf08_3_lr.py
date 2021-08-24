# 실습 
# 1. [4]
# 2. [5, 6]
# 3. [6, 7, 8]

import tensorflow as tf
tf.set_random_seed(55)

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

optimizer = tf.train.AdamOptimizer(learning_rate=0.6221792)
train = optimizer.minimize(loss) # loss의 최소값을 찾아줌

#### 그래프 형태로 만들어줌 출력하려면 sessrun에 넣어줘야함
# sess = tf.Session()

sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 출력하기전에 gvi 해줌

for step in range(101): # -> 2000번 epochs
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

'''
0 61.770924 [-0.78622025] [1.0887307]
20 3.8311005 [0.7369626] [1.852822]
40 0.26558438 [1.4767987] [1.5970086]
60 0.03617185 [1.8061541] [1.2195739]
80 0.008331689 [1.9471438] [1.0223818]
100 0.0008315868 [1.9999999] [0.9811333]
[array([8.9811325], dtype=float32)]
[array([10.9811325, 12.9811325], dtype=float32)]
[array([12.9811325, 14.9811325, 16.981133 ], dtype=float32)]

'''
