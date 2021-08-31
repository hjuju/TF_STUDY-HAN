# from keras import datasets
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.set_random_seed(66)

### get_variable vs Variable
### CNN, activation


# 1. 데이터
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

learning_rate = 0.00001
training_epochs = 3
batch_size = 100
total_batch = int(len(x_train)/batch_size)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])



# 모델구성
# 변수 초기화
##### tensorflow 1
## layer 1
w1 = tf.get_variable('w1', shape=[3, 3, 1, 32])   # [kernel_size, input의 channel(color) 수(맨 마지막), output]
layer1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')   # x, w의 차원이 같아야 함(둘이 연산되어야 하니까)    # strides : 4차원으로 잡아줘야 함
layer1_activation = tf.nn.relu(layer1)     # activation
layer1_maxpool = tf.nn.max_pool(layer1_activation, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')     # maxpool
                                                  # (자리채우기 위함, kenelsize:(2,2), 자리채우기 위함)
##### tensorflow 2
# model = Sequential()                                                                                                  color
# model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same', input_shape=(28, 28, 1),    # (low, cols, channel)
#                  activation='relu'))

print(layer1_activation)    # (?, 28, 28, 32)
print(layer1_maxpool)       # (?, 14, 14, 32)



## layer 2
w2 = tf.get_variable('w2', shape=[3, 3, 32, 64])
layer2 = tf.nn.conv2d(layer1_maxpool, w2, strides=[1,1,1,1], padding='SAME')
layer2_activation = tf.nn.selu(layer2)
layer2_maxpool = tf.nn.max_pool(layer2_activation, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(layer2_activation)    # (?, 14, 14, 64)
print(layer2_maxpool)       # (?, 7, 7, 64)




## layer 3
w3 = tf.get_variable('w3', shape=[3, 3, 64, 128])
layer3 = tf.nn.conv2d(layer2_maxpool, w3, strides=[1,1,1,1], padding='SAME')
layer3_activation = tf.nn.elu(layer3)
layer3_maxpool = tf.nn.max_pool(layer3_activation, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(layer3_activation)    # (?, 7, 7, 128)
print(layer3_maxpool)       # (?, 4, 4, 128)




## layer 4
w4 = tf.get_variable('w4', shape=[2, 2, 128, 64], initializer=tf.contrib.layers.xavier_initializer())       # 가중치초기화 : 가중치폭발을 막기 위해서
layer4 = tf.nn.conv2d(layer3_maxpool, w4, strides=[1,1,1,1], padding='VALID')
layer4_activation = tf.nn.leaky_relu(layer4)
layer4_maxpool = tf.nn.max_pool(layer4_activation, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(layer4_activation)    # (?, 3, 3, 64)
print(layer4_maxpool)       # (?, 2, 2, 64)




## Flatten
L_flat = tf.reshape(layer4_maxpool, [-1, 2*2*64])
print("플랫튼 :", L_flat)    # (?, 256)




## layer5  DNN
w5 = tf.get_variable('w5', shape=[2*2*64, 64], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([64]), name='b1')
layer5 = tf.matmul(L_flat, w5) + b5
layer5_activation = tf.nn.selu(layer5)
layer5_dropout = tf.nn.dropout(layer5_activation, keep_prob=0.2)

print(layer5_dropout)        # (?, 64)




## layer6  DNN
w6 = tf.get_variable('w6', shape=[64, 32], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([32]), name='b2')
layer6 = tf.matmul(layer5_dropout, w6) + b6
layer6_activation = tf.nn.selu(layer6)
layer6_dropout = tf.nn.dropout(layer6_activation, keep_prob=0.2)

print(layer6_dropout)        # (?, 32)




## layer7  softmax
w7 = tf.get_variable('w7', shape=[32, 10])
b7 = tf.Variable(tf.random_normal([10]), name='b3')
layer7 = tf.matmul(layer6_dropout, w7) + b7
hypothesis = tf.nn.softmax(layer7)

print(hypothesis)      # (?, 10)


# 3. 컴파일 , 훈련
# categorical_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))       

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
# train = optimizer.minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)  # 위랑 동일
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_loss= 0

    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size
        batch_x, batch_y = x_train[start:end], y_train[start:end]

        feed_dict = {x:batch_x, y:batch_y}

        batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)

        avg_loss += batch_loss/total_batch

    print('Epoch :', '%04d' %(epoch + 1), 'loss : {:.9f}'.format(avg_loss))     
                                                    # 소숫점9자리까지
print("훈련 끝!!!")

prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('ACC :', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))


# 0.996 이상