import tensorflow as tf
print(tf.__version__)

# print("helloworld")

hello = tf.constant("helloworld") # hello안에는 helloworld의 상수만 들어있음 --> 인풋
print(hello)
# Tensor("Const:0", shape=(), dtype=string) sess.run을 하지 않으면 자료구조가 출력 됨, hello변수의 내용을 보고 싶으면 Session에 넣어줘야함


sees = tf.Session() # 변수든 상수든 세션에 넣어서 사용해야함
print(sees.run(hello)) # --> 변수

# b'helloworld'