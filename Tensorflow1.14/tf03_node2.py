# 실습 
# 덧셈 node3
# 뺄셈 node4
# 곱셈 node5
# 나눗셈 node6

import tensorflow as tf
from tensorflow.python.client.session import Session
sess = Session()
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)
node3 = tf.add(node1, node2)
node4 = tf.subtract(node1, node2)
node5 = tf.multiply(node1, node2)
node6 = tf.divide(node1, node2)

print('add: ',sess.run(node3))
print('substract: ', sess.run(node4))
print('multiply: ',sess.run(node5))
print('divide: ',sess.run(node6))

'''
add:  5.0
substract:  -1.0
multiply:  6.0
divide:  0.6666667

'''