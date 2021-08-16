import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import pandas as pd

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

from glob import glob 


def get_input_dataset(data, index, train = False) : 
    input0 = tf.convert_to_tensor(data[0][index].toarray(), tf.float32)
    input1 = tf.convert_to_tensor(data[1][index].toarray(), tf.float32)
    input2 = tf.convert_to_tensor(data[2][index].toarray(), tf.float32)
    
    if train : 
        label = labels[index]

        return input0, input1, input2, label
    else:
        return input0, input1, input2,

def single_dense(x, units):
    fc = Dense(units, activation = None, kernel_initializer = 'he_normal')(x)
    batch = BatchNormalization()(fc)
    relu = ReLU()(batch)
    dr = Dropout(0.2)(relu)
    
    return dr

def create_model(input_shape0,input_shape1,input_shape2, num_labels, learning_rate):
    x_in0 = Input(input_shape0,)
    x_in1 = Input(input_shape1,)
    x_in2 = Input(input_shape2,)
    
    fc0 = single_dense(x_in0, 512)
    fc0 = single_dense(fc0, 256)
    fc0 = single_dense(fc0, 128)
    fc0 = single_dense(fc0, 64)
    
    # fc1 = single_dense(x_in1, 1024)
    fc1 = single_dense(x_in1, 512)
    fc1 = single_dense(fc1, 256)
    fc1 = single_dense(fc1, 128)
    fc1 = single_dense(fc1, 64)
    
    fc2 = single_dense(x_in2, 512)
    fc2 = single_dense(fc2, 256)
    fc2 = single_dense(fc2, 128)
    fc2 = single_dense(fc2, 64)
    
    fc = Concatenate()([fc0,fc1,fc2])
    
    # fc = single_dense(fc, 128)
    fc = single_dense(fc, 64)
    
    x_out = Dense(num_labels, activation = 'softmax')(fc)
    
    model = Model([x_in0,x_in1,x_in2], x_out)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
    
    return model

with open('./Dacon/_save/npy/climate/pkl/inputs.pkl','rb') as f :
    train_inputs, test_inputs, labels = pickle.load(f)


num_labels = 46
learning_rate = 5e-2
seed = np.random.randint(2**16-1)
skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = seed)

for train_idx, valid_idx in skf.split(train_inputs[0], labels):
    X_train_input0, X_train_input1, X_train_input2, X_train_label = get_input_dataset(train_inputs, train_idx, train = True)
    X_valid_input0, X_valid_input1, X_valid_input2, X_valid_label = get_input_dataset(train_inputs, valid_idx, train = True)
    
    now = datetime.datetime.now()
    now = str(now)[11:16].replace(':','h')+'m'
    ckpt_path = f'./Dacon/_save/MCP/climate/{now}.ckpt'
    
    input_shape0 = X_train_input0.shape[1]
    input_shape1 = X_train_input1.shape[1]
    input_shape2 = X_train_input2.shape[1]


    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor = 'val_acc', save_best_only= True, save_weights_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor=0.7, patience = 2,),
                ]
    model = create_model(input_shape0,input_shape1,input_shape2, num_labels, learning_rate)
    model.fit(
                        [X_train_input0,X_train_input1,X_train_input2],
                        X_train_label,
                        epochs=1000,
                        callbacks=callbacks,
                        validation_data=([X_valid_input0, X_valid_input1, X_valid_input2], X_valid_label),
                        verbose=1,  # Logs once per epoch.
                        batch_size=1024)
    
    model.load_weights(ckpt_path)
    prediction = model.predict([test_inputs[0], test_inputs[1], test_inputs[2]])
    np.save(f'./Dacon/_save/MCP/climate/climate{now}_prediction.npy', prediction)


predictions = []
for ar in glob('./Dacon/_save/MCP/climate/climate01h03m_prediction.npy'):
    arr = np.load(ar)
    predictions.append(arr)

sample = pd.read_csv('./Dacon/_data/climate/sample_submission.csv')
sample['label'] = np.argmax(np.mean(predictions,axis=0), axis = 1)
date_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
sample.to_csv('./Dacon/_save/csv/climate/predict' + date_time + '.csv', index=False)

with open('./Dacon/_save/npy/climate/pkl/inputs.pkl','rb') as f :
    train_inputs, test_inputs, labels = pickle.load(f)
