#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Rafael Teixeira'
__version__ = '0.1'
__email__ = 'rafaelgteixeira@ua.pt'
__status__ = 'Development'

import argparse
import gc
import json
import pathlib
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

tf.keras.utils.set_random_seed(42)

import keras.backend as K

def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
#
#def mcc_loss(y_true, y_pred):
#    return 1.0-matthews_correlation(y_true, y_pred)

def mcc_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0) * 1e2
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0) / 1e2
    
    up = tp*tn - fp*fn
    down = K.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
    
    mcc = up / (down + K.epsilon())
    mcc = tf.where(tf.math.is_nan(mcc), tf.zeros_like(mcc), mcc)
    
    return 1 - K.mean(mcc)


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, start_time):
        super().__init__()
        self.validation_data = validation_data
        self.start_time = start_time

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_mccs = []
        self.times = []
    
    def on_epoch_end(self, epoch, logs=None):
        predictions = [0 if x < 0.5 else 1 for x in self.model.predict(self.validation_data[0], verbose=0)]

        val_f1 = f1_score(self.validation_data[1], predictions)
        val_mcc = matthews_corrcoef(self.validation_data[1], predictions)

        self.val_f1s.append(val_f1)
        self.val_mccs.append(val_mcc)
        self.times.append(time.time() - self.start_time)

        print("- val_f1: %f - val_mcc %f" %(val_f1, val_mcc))

        tf.keras.backend.clear_session()
        gc.collect()

    def get_metrics(self):
        return self.val_f1s, self.val_mccs, self.times

def create_LSTM(look_back, n_features):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.GRU(256, input_shape=(look_back, n_features), return_sequences=True))
    model.add(tf.keras.layers.GRU(128))
    #model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy', matthews_correlation])

    return model

def train_LR(input, output):

    input = pathlib.Path(input)

    output = pathlib.Path(output)

    output.mkdir(parents=True, exist_ok=True)

    X_train = np.loadtxt(input/"x_train.csv", delimiter=",")

    y_train = np.loadtxt(input/"y_train.csv", delimiter=",", dtype=int)

    X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = LogisticRegression(random_state=42, C=0.1, max_iter=1000).fit(X_train, y_train)

    predictions = model.predict(X_cv)

    predictions = [0 if x < 0.5 else 1 for x in predictions]
    print(confusion_matrix(y_cv, predictions))
    print("F1:", f1_score(y_cv, predictions))
    print("MCC:", matthews_corrcoef(y_cv, predictions))


def train_LSTM(input, output, look_back):
    start = time.time()

    input = pathlib.Path(input)

    output = pathlib.Path(output)

    output.mkdir(parents=True, exist_ok=True)

    X_train = np.loadtxt(input/"x_train.csv", delimiter=",")
    X_train = np.reshape(X_train, (X_train.shape[0], look_back, -1))

    y_train = np.loadtxt(input/"y_train.csv", delimiter=",", dtype=float)
    print(y_train.shape, sum(y_train))
    X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = create_LSTM(look_back, X_train.shape[-1])

    val_dataset = tf.data.Dataset.from_tensor_slices((X_cv, y_cv)).batch(1024)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(1024)

    custom_metrics = CustomCallback((val_dataset, y_cv), start)
    history = model.fit(train_dataset,
            epochs=300, verbose = 2, 
            callbacks=[custom_metrics])
    
    metrics = custom_metrics.get_metrics()

    history.history["f1"] = metrics[0]
    history.history["mcc"] = metrics[1]
    history.history["times"] = metrics[2]

    history = json.dumps(history.history)
    
    f = open(output/"train_history.json", "w")
    f.write(history)
    f.close()

    model.save(output/'trained_model.h5')

    predictions = model.predict(X_cv)

    predictions = [0 if x < 0.5 else 1 for x in predictions]
    print(confusion_matrix(y_cv, predictions))
    print("F1:", f1_score(y_cv, predictions))
    print("MCC:", matthews_corrcoef(y_cv, predictions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test the model')
    parser.add_argument('-f', type=str, help='Processed dataset folder', default='dataset/one_hot_encoding/')
    parser.add_argument('-o', type=str, help='Output folder', default='results/')
    parser.add_argument('-t', type=str, help='Training type (a)all, (m)LSTM, l(LR)', default="a")
    parser.add_argument('-l', type=int, help='Model look back', default=10)

    args = parser.parse_args()
    if args.t == "a" or args.t == "m":
        train_LSTM(args.f, args.o, args.l)
    if args.t == "a" or args.t == "l":
        train_LR(args.f, args.o)
