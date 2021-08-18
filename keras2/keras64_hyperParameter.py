import imp
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D
from tensorflow.python.keras.backend import dropout
from icecream import ic

(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255

#2 모델
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')

    return model

def create_hyperparameter():
    batches = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {"batch_size": batches, "optimizer" : optimizers, "drop": dropout}

hyperparameters = create_hyperparameter()
# ic(hyperparameters)
# model2 = build_model()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1) # 텐서플로를 사이킷런에 wrapping

model = RandomizedSearchCV(model2, hyperparameters, cv=5) # 서치 모델에 텐서플로 모델 입력안됨 -> 텐서플로모델을 사이킷런으로 wrapping

model.fit(x_train, y_train, verbose=1)

be = model.best_estimator_
bp = model.best_params_
bs = model.best_score_

print("best_estimator:", be)
print("best_params: ", bp)
print("best_score", bs)

