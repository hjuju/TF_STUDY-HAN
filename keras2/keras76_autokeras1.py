# pip install autokeras  /  python 3.8.8  / tensorflow 2.4.1

import autokeras as ak

from tensorflow.keras.datasets import mnist


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 모델
model = ak.ImageClassifier(overwrite=True, max_trials=1)        # max_trials : 모델을 2번 돌리겠다.   # validation default:0.2

# 3. 훈련
model.fit(x_train, y_train, epochs=2)

# 4. 평가, 예측
y_predict = model.predict(x_test)

results = model.evaluate(x_test, y_test)
print(results)

# [0.04554564133286476, 0.9840999841690063]  loss / accuracy