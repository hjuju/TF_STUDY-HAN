import tensorflow as tf

tf.compat.v1.set_random_seed(777)

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = x * W + b

#### 그래프 형태로 만들어줌 출력하려면 sessrun에 넣어줘야함
# sess = tf.Session()

sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 출력하기전에 gvi 해줌

sess = tf.InteractiveSession() 
sess.run(tf.global_variables_initializer()) # 초기화
bbb = hypothesis.eval() # 변수.eval / sess.run(w)와 똑같음(방식차이)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = hypothesis.eval(session=sess)
print("ccc: ", ccc)
sess.close()
