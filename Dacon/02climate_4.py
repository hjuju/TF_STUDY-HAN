import pandas as pd
import os
import random
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import matplotlib.pyplot as plt

import numpy as np

# 데이터를 불러오는 방법
def load_dataset(path, seed = 20210803) : 
    data_path = os.path.join(path, 'train.csv')
    data = pd.read_csv(data_path)
    data_texts = data['사업명'] + data['과제명'] + data['요약문_연구목표'].fillna('.').astype('str')
    data_labels = data['label']
    
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    
    for train_idx, valid_idx in skf.split(data_texts, data_labels):
        train_texts = data_texts.iloc[train_idx]
        train_labels = data_labels.iloc[train_idx]
        
        valid_texts = data_texts.iloc[valid_idx]
        valid_labels = data_labels.iloc[valid_idx]
        break;
    
    
    return  ((data_texts, data_labels),
            (train_texts, train_labels),
            (valid_texts, valid_labels))

def train_ngram_vectorize(top_k) : 
    kwargs = {
        'ngram_range' : (1,2),
        'dtype' : 'int32',
        'strip_accents' : False,
        'lowercase' : False,
        'decode_error' : 'replace',
        'analyzer': 'word',
        'min_df' : 2,
        
    }
    
    vectorizer = TfidfVectorizer(**kwargs)
    
    x_train = vectorizer.fit_transform(train_set[0])
    x_val = vectorizer.transform(valid_set[0])
    
    selector = SelectKBest(f_classif, k=min(top_k, x_train.shape[1]))
    selector.fit(x_train, train_set[1].values)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    
    return x_train, x_val


def single_dense(x, units):
    fc = Dense(units, activation = None, kernel_initializer = 'he_normal')(x)
    batch = BatchNormalization()(fc)
    relu = ReLU()(batch)
    dr = Dropout(0.2)(relu)
    
    return dr

def create_model(input_shape, num_labels, learning_rate):
    x_in = Input(input_shape,)
    unit = 512
    fc = single_dense(x_in, unit)
    while unit > 64 :
        unit = unit // 2
        fc = single_dense(fc, unit)
    
    x_out = Dense(num_labels, activation = 'softmax')(fc)
    
    model = Model(x_in, x_out)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
    
    return model

def get_valid_score(x_train, x_val, top_k):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                ]
    model = create_model(x_train.shape[1], 46, 1e-3, top_k)
    history = model.fit(
                        x_train.toarray(),
                        train_set[1],
                        epochs=1000,
                        callbacks=callbacks,
                        validation_data=(x_val.toarray(), valid_set[1]),
                        verbose=0,  # Logs once per epoch.
                        batch_size=1024)

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(history.history['loss'], 'y', label='train loss')
    loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    acc_ax.plot(history.history['acc'], 'b', label='train acc')
    acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper left')

    plt.show()
    print(np.argmin(history.history['val_loss']))
    print(max(history.history['val_acc']))
    
    print('='*50)

def test_ngram_vectorize(top_k) : 
    kwargs = {
        'ngram_range' : (1,2),
        'dtype' : 'int32',
        'strip_accents' : False,
        'lowercase' : False,
        'decode_error' : 'replace',
        'analyzer': 'word',
        'min_df' : 2,
        
    }
    
    vectorizer = TfidfVectorizer(**kwargs)
    
    x_train = vectorizer.fit_transform(data_set[0])
    x_test = vectorizer.transform(test_set)
    
    selector = SelectKBest(f_classif, k=min(top_k, x_train.shape[1]))
    selector.fit(x_train, data_set[1].values)
    x_train = selector.transform(x_train).astype('float32')
    x_test = selector.transform(x_test).astype('float32')
    
    return x_train, x_test


def plot_loss(history):
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(history.history['loss'], 'y', label='train loss')
    loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    acc_ax.plot(history.history['acc'], 'b', label='train acc')
    acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper left')

    plt.show()


def modeling(seed):
    print(f"its seed is {seed}")
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    predictions = []
    skf_score = []
    for train_idx, valid_idx in skf.split(data_set[0], data_set[1]):
        X_train_texts = x_train[train_idx]
        X_train_labels = data_set[1].iloc[train_idx]

        X_valid_texts = x_train[valid_idx]
        X_valid_labels = data_set[1].iloc[valid_idx]

        model = create_model(x_train.shape[1], 46, 1e-3)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
            tf.keras.callbacks.ModelCheckpoint('./model.ckpt', monitor = 'val_acc', save_best_only= True, save_weights_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor=0.9, patience = 3,),

                    ]
        history = model.fit(
                            X_train_texts.toarray(),
                            X_train_labels,
                            validation_data = (X_valid_texts.toarray(), X_valid_labels),
                            epochs=1000,
                            callbacks = callbacks,
                            verbose=0,  # Logs once per epoch.
                            batch_size=1024)

        model.load_weights('./model.ckpt')
        prediction = model.predict(x_test)
        predictions.append(prediction)
        skf_score.append(max(history.history['val_acc'])) 
        print('Validation Done')
        print('='*50)
    print('=== VALIDATION DONE ===')
    print('VALIDATION SCORE : ', np.mean(skf_score))
    p = np.argmax(np.mean(predictions,axis = 0), axis = 1)
    s = np.mean(skf_score)
    sample = pd.read_csv('./Dacon/_data/climate/sample_submission.csv')
    sample['label'] = p
    print(sample.head())
    sample.to_csv(f'Day0804_JayHong_seed_{seed}_{round(s,5)}.csv',index=False)
    pred = [predictions, s]
    
    return pred

path = './Dacon/_data/climiate/'
data_set, train_set, valid_set = load_dataset(path, seed = 20210804)
test = pd.read_csv('test.csv')
test_set = test['사업명'] + test['과제명'] + test['요약문_연구목표'].fillna('.').astype('str')
Ngram_range = (1,2)
top_k = 80000 # 최대 8만 단어 사용

x_train, x_test = test_ngram_vectorize(top_k)

path = './Dacon/_data/climiate/'
data_set, train_set, valid_set = load_dataset(path, seed = 20210804)
test = pd.read_csv('test.csv')
test_set = test['사업명'] + test['과제명'] + test['요약문_연구목표'].fillna('.').astype('str')
Ngram_range = (1,2)
top_k = 80000 # 최대 8만 단어 사용

x_train, x_test = test_ngram_vectorize(top_k)

total_preds = []
for i in range(10):
    print(f'===== EPOCH {i} STARTED =====')
    seed = random.randint(0, 2**16-1)
    pred = modeling(seed)
    total_preds.append(pred)
    print('=== EPOCH DONE === ')

import pandas as pd
from glob import glob
import numpy as np

submission_files = glob('Day0804*.csv')
submission_files = sorted(submission_files, key = lambda x: x.split('0.')[1].split('.csv')[0], reverse = False)

submits = []
for sub_file in submission_files:
    submit = pd.read_csv(sub_file)['label']
    submits.append(submit)
final_pred = pd.DataFrame(submits).mode().T
final_pred = final_pred.fillna('None')
final_pred = final_pred.apply(lambda x : x[0] if x[1] == 'None' else x[1], axis= 1).values

sample = pd.read_csv('./Dacon/_data/climate/sample_submission.csv')
sample['label'] = final_pred

import datetime
date_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
sample.to_csv('./Dacon/_save/csv/climate/predict' + date_time + '.csv', index=False)
sample.head()