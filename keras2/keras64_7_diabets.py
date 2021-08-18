import imp
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_boston, load_wine
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D
from tensorflow.python.keras.backend import dropout
from icecream import ic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.python.keras.layers.core import Flatten

#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2 모델
def build_model(drop=0.5,opt=Adam,lr=0.1):
    inputs = Input(shape=(4,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(3, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=opt(learning_rate=lr), metrics=['acc'], loss='categorical_crossentropy')

    return model

def create_hyperparameter():
    batches = [128,256,512,1028]
    optimizers = [Adam, Adadelta]
    dropout = [0.1, 0.2, 0.3]
    lr = [0.5, 0.01, 0.001]
    return {"batch_size": batches, "opt" : optimizers, "drop": dropout, "lr" : lr}

hyperparameters = create_hyperparameter()
# ic(hyperparameters)
# model2 = build_model()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.05) 

model2 = KerasRegressor(build_fn=build_model, verbose=1) # 텐서플로를 사이킷런에 wrapping

model = RandomizedSearchCV(model2, hyperparameters, cv=5) # 서치 모델에 텐서플로 모델 입력안됨 -> 텐서플로모델을 사이킷런으로 wrapping

model.fit(x_train, y_train, verbose=1, epochs=3, validation_split=0.2) # 파라미터가 우선순위로 적용됨

be = model.best_estimator_
bp = model.best_params_
bs = model.best_score_

print("best_estimator:", be)
print("best_params: ", bp)
print("best_score", bs)
acc = model.score(x_test,y_test)
y_pred = model.predict(x_test)
ic(acc)

