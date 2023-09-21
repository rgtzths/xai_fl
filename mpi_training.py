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

from mpi4py import MPI
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

tf.keras.utils.set_random_seed(42)


def create_model(look_back, n_features):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(2048,input_shape=(look_back, n_features)))
    model.add(tf.keras.layers.Dense(1024, activation="tanh"))
    model.add(tf.keras.layers.Dense(512, activation="tanh"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

    return model

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser(description='Train and test the model')
parser.add_argument('--g_epochs', type=int, help='Global epochs number', default=10)
parser.add_argument('--l_epochs', type=int, help='local epochs number', default=1)
parser.add_argument('-b', type=int, help='Batch size', default=256)
parser.add_argument('-d', type=str, help='Dataset', default="dataset/temp_dataset")
parser.add_argument('-o', type=str, help='Output folder', default="results")
parser.add_argument('-l', type=int, help='Lookback used', default=10)

args = parser.parse_args()

global_epochs = args.g_epochs
local_epochs = args.l_epochs
batch_size = args.b
dataset = args.d
output = args.o
look_back = args.l

output = pathlib.Path(output)
output.mkdir(parents=True, exist_ok=True)
dataset = pathlib.Path(dataset)


start = time.time()

if rank == 0:
    node_weights = []
    X_cv = np.loadtxt(dataset/"x_cv.csv", delimiter=",")
    X_cv = np.reshape(X_cv, (X_cv.shape[0], look_back, -1))

    y_cv = np.loadtxt(dataset/"y_cv.csv", delimiter=",", dtype=int)

    val_dataset = tf.data.Dataset.from_tensor_slices(X_cv).batch(batch_size)

    model = create_model(look_back, X_cv.shape[-1])

    #Get the amount of training examples of each worker and divides it by the total
    #of examples to create a weighted average of the model weights
    for node in range(1, size):
        status = MPI.Status()
        node_weights.append(comm.recv(source=node, tag=1000, status=status))
    
    total_size = sum(node_weights)

    node_weights = [weight/total_size for weight in node_weights]
    results = {"acc" : [], "mcc" : [], "f1" : [], "times" : {"epochs" : [], "loads" : []}}
    results["times"]["loads"].append(time.time() - start)

else:
    X_train = np.loadtxt(dataset/("x_train_subset_%d.csv" % rank), delimiter=",")
    X_train = np.reshape(X_train, (X_train.shape[0], look_back, -1))

    y_train = np.loadtxt(dataset/("y_train_subset_%d.csv" % rank), delimiter=",", dtype=int)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)

    model = create_model(look_back, X_train.shape[-1])

    comm.send(len(X_train), dest=0, tag=1000)

model.set_weights(comm.bcast(model.get_weights(), root=0))
if rank == 0:
    results["times"]["loads"].append(time.time() - start)

for global_epoch in range(global_epochs):

    avg_weights = None
    if rank == 0:
        print("\nStart of epoch %d" % global_epoch)
        for node in range(1, size):
            weights = comm.recv(source=node, tag=global_epoch)

            if node == 1:
                avg_weights = [ weight * node_weights[node-1] for weight in weights]
            else:
                avg_weights = [ avg_weights[i] + weights[i] * node_weights[node-1] for i in range(len(weights))]
        
    else:
        model.fit(train_dataset, epochs=local_epochs, verbose=0)
        comm.send(model.get_weights(), dest=0, tag=global_epoch)

    avg_weights = comm.bcast(avg_weights, root=0)
    
    model.set_weights(avg_weights)

    if rank == 0:
        predictions = [np.argmax(x) for x in model.predict(val_dataset, verbose=0)]
        train_f1 = f1_score(y_cv, predictions, average="macro")
        train_mcc = matthews_corrcoef(y_cv, predictions)
        train_acc = accuracy_score(y_cv, predictions)

        results["acc"].append(train_acc)
        results["f1"].append(train_f1)
        results["mcc"].append(train_mcc)
        results["times"]["epochs"].append(time.time() - start)
        print("- val_f1: %f - val_mcc %f - val_acc %f" %(train_f1, train_mcc, train_acc))
            

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()

if rank==0:
    history = json.dumps(results)

    f = open( output/"train_history.json", "w")
    f.write(history)
    f.close()

    model.save(output/'trained_model.h5')