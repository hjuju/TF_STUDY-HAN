from sklearn.datasets import load_breast_cancer
import tensorflow as tf
tf.set_random_seed(66)

datasets = load_breast_cancer()

x_data = datasets.data
y_data = datasets.target
print(x_data.shape, y_data.shape)     # (569, 30) (569,)



# 2. 모델 구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, ])

W = tf.compat.v1.Variable(tf.random.normal([30,1]), name='weight')       # x_data(5, 3) * w(3, 1) = hypothesis(5, 1)
b = tf.compat.v1.Variable(tf.random.normal([1]), name='bias')                      # 3      3 동일해야 함 
'''
### 행렬의 곱
(a, b) * (c, d)
  => b랑 c의 크기가 동일해야 곱할 수 있음
결과값 shape : (a, d)
'''

# hypothesis = x * W + b            # 1차원일 때
# hypothesis = tf.matmul(x, W) + b    # 다차원일 때(행렬곱)
hypothesis = tf.sigmoid(tf.matmul(x, W) + b)   # 2진분류     # activation함수 : 출력되는 값을 tf.sigmoid로 감쌈

# cost = tf.reduce_mean(tf.square(hypothesis-y))      # mse
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))     # binary_crossentropy :  y 값이 0과 1 사이로 바꼈으니까

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())


# 3. 훈련
for epochs in range(5):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                        feed_dict={x:x_data, y:y_data})
    if epochs % 20 == 0:        # 10번에 1번씩 출력
        print(epochs, "cost :", cost_val, "\n", "hy_val :", hy_val)


# 4. 평가, 예측
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) # 0.5보다 크면 1, 작으면 0      # cast : 조건에 부합하면 1, 부합하지 않으면 0
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))    # reduce_mean : 평균

h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x:x_data, y:y_data})

print("========================================================")
print(f'predict value : {h[0:5]} \n "original value: \n{c[0:5]} \naccuracy: : {a}')

sess.close()