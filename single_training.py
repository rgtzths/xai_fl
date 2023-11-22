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
from sklearn.ensemble import RandomForestClassifier

tf.keras.utils.set_random_seed(42)

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data,  train_data, start_time, verbose=0, checkpoint_file="model_checkpoint.keras"):
        super().__init__()
        self.X_val = validation_data[0]
        self.y_val = validation_data[1]
        self.X_train = train_data[0]
        self.y_train = train_data[1]

        self.start_time = start_time
        self.verbose = verbose
        self.best_mcc = -1
        self.checkpoint_file = checkpoint_file

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_mccs = []
        self.train_f1s = []
        self.train_mccs = []
        self.times = []
    
    def on_epoch_end(self, epoch, logs=None):
        predictions = [round(x[0]) for x in self.model.predict(self.X_val, verbose=0, batch_size=8192)]

        val_f1 = f1_score(self.y_val, predictions)
        val_mcc = matthews_corrcoef(self.y_val, predictions)

        if val_mcc > self.best_mcc:
            self.best_mcc = val_mcc
            self.model.save_weights(self.checkpoint_file, overwrite=True, save_format=None, options=None)
        
        self.val_f1s.append(val_f1)
        self.val_mccs.append(val_mcc)
        self.times.append(time.time() - self.start_time)

        predictions = [round(x[0]) for x in self.model.predict(self.X_train, verbose=0, batch_size=1024)]

        train_f1 = f1_score(self.y_train, predictions)
        train_mcc = matthews_corrcoef(self.y_train, predictions)
        self.train_f1s.append(train_f1)
        self.train_mccs.append(train_mcc)
        if self.verbose > 0:
            print("- val_f1: %f - val_mcc %f - train_f1: %f - train_mcc %f" %(val_f1, val_mcc, train_f1, train_mcc))

        tf.keras.backend.clear_session()
        gc.collect()

    def get_metrics(self):
        return self.train_f1s, self.train_mccs, self.val_f1s, self.val_mccs, self.times

def create_LSTM(look_back, n_features):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(512, input_shape=(look_back, n_features), activation="tanh"))
    model.add(tf.keras.layers.Dense(512, activation="tanh"))
    model.add(tf.keras.layers.Dense(256, activation="tanh"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    return model

def create_LSTM2(look_back, n_features):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(256, input_shape=(look_back, n_features), return_sequences=True))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True, activation="relu"))
    model.add(tf.keras.layers.LSTM(64, return_sequences=True, activation="relu"))
    model.add(tf.keras.layers.LSTM(32, return_sequences=True, activation="relu"))
    model.add(tf.keras.layers.LSTM(32, return_sequences=True, activation="relu" ))
    model.add(tf.keras.layers.LSTM(64, return_sequences=True, activation="relu"))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True, activation="relu"))
    model.add(tf.keras.layers.LSTM(256, activation="relu"))
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    return model

def train_LR(input, output):

    input = pathlib.Path(input)

    output = pathlib.Path(output)

    output.mkdir(parents=True, exist_ok=True)

    X_train = np.loadtxt(input/"x_train.csv", delimiter=",")
    y_train = np.loadtxt(input/"y_train.csv", delimiter=",")

    X_cv = np.loadtxt(input/"x_cv.csv", delimiter=",")
    y_cv = np.loadtxt(input/"y_cv.csv", delimiter=",")

    model = LogisticRegression(random_state=42, C=0.1, max_iter=1000).fit(X_train, y_train)

    predictions = model.predict(X_cv)

    predictions = [0 if x < 0.5 else 1 for x in predictions]
    print("LR results: ")
    print(confusion_matrix(y_cv, predictions))
    print("F1:", f1_score(y_cv, predictions))
    print("MCC:", matthews_corrcoef(y_cv, predictions))

def train_RF(input, output):

    input = pathlib.Path(input)

    output = pathlib.Path(output)

    output.mkdir(parents=True, exist_ok=True)

    X_train = np.loadtxt(input/"x_train.csv", delimiter=",")
    y_train = np.loadtxt(input/"y_train.csv", delimiter=",")

    X_cv = np.loadtxt(input/"x_cv.csv", delimiter=",")
    y_cv = np.loadtxt(input/"y_cv.csv", delimiter=",")

    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)

    predictions = model.predict(X_cv)

    predictions = [0 if x < 0.5 else 1 for x in predictions]
    print("RF results: ")
    print(confusion_matrix(y_cv, predictions))
    print("F1:", f1_score(y_cv, predictions))
    print("MCC:", matthews_corrcoef(y_cv, predictions))

def train_LSTM(input, output, look_back, verbose=0):
    start = time.time()

    input = pathlib.Path(input)

    output = pathlib.Path(output)

    output.mkdir(parents=True, exist_ok=True)

    X_train = np.loadtxt(input/"x_train.csv", delimiter=",")

    y_train = np.loadtxt(input/"y_train.csv", delimiter=",").reshape((-1,1))

    X_cv = np.loadtxt(input/"x_cv_under.csv", delimiter=",")
    y_cv = np.loadtxt(input/"y_cv_under.csv", delimiter=",").reshape((-1,1))

    X_train = np.reshape(X_train, (X_train.shape[0], look_back, -1))
    X_cv = np.reshape(X_cv, (X_cv.shape[0], look_back, -1))

    model = create_LSTM(look_back, X_train.shape[-1])

    custom_metrics = CustomCallback((X_cv, y_cv), (X_train, y_train), start, verbose, output/"checkpoint_file.keras")

    history = model.fit(X_train, y_train,
            epochs=50, verbose = verbose, batch_size=1024,
            callbacks=[custom_metrics])
    
    metrics = custom_metrics.get_metrics()

    history.history["f1_train"] = metrics[0]
    history.history["mcc_train"] = metrics[1]
    history.history["f1_val"] = metrics[2]
    history.history["mcc_val"] = metrics[3]
    history.history["times"] = metrics[4]

    history = json.dumps(history.history)
    
    f = open(output/"train_history.json", "w")
    f.write(history)
    f.close()

    model.load_weights(output/"checkpoint_file.keras")

    predictions = model.predict(X_cv, verbose=verbose)

    predictions = [ round(x[0]) for x in predictions]
    print("LSTM results: ")
    print(confusion_matrix(y_cv, predictions))
    print("F1:", f1_score(y_cv, predictions))
    print("MCC:", matthews_corrcoef(y_cv, predictions))
    model.save(output/'trained_model.keras')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test the model')
    parser.add_argument('-f', type=str, help='Processed dataset folder', default='dataset/one_hot_encoding/')
    parser.add_argument('-o', type=str, help='Output folder', default='results/')
    parser.add_argument('-t', type=str, help='Training type (a)all, (m)LSTM, (l)LR (r)RF', default="a")
    parser.add_argument('-l', type=int, help='Model look back', default=10)

    args = parser.parse_args()
    if args.t == "a" or args.t == "l":
        print("Training LR")
        train_LR(args.f, args.o)
    if args.t == "a" or args.t == "r":
        print("Training RF")
        train_RF(args.f, args.o)
    if args.t == "a" or args.t == "m":
        print("Training LSTM")
        train_LSTM(args.f, args.o, args.l, 2)
    