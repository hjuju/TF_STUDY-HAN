import tensorflow as tf
sess = tf.Session()

x = tf.Variable([2], dtype=tf.float32, name='test')

init = tf.global_variables_initializer() # 변수 초기화 됨 그래프에 들어가기에 적합한 상태가 됨

sess.run(init)
print(sess.run(x))
