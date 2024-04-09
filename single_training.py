# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse
import tensorflow as tf
import numpy as np
import time
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, matthews_corrcoef

import json
import pathlib
from config import DATASETS, OPTIMIZERS

tf.keras.utils.set_random_seed(42)

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val,  y_val, start_time, verbose=0, checkpoint_file="model_checkpoint.keras", threshold=1, patience=10, min_delta=0.001):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val

        self.start_time = start_time
        self.verbose = verbose
        self.best_mcc = -1
        self.checkpoint_file = checkpoint_file
        self.threshold = threshold
        self.patience_buffer = [0]*patience
        self.min_delta = min_delta

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_mccs = []
        self.train_f1s = []
        self.train_mccs = []
        self.times = []
    
    def on_epoch_end(self, epoch, logs=None):
        predictions = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)

        val_f1 = f1_score(self.y_val, predictions, average="macro")
        val_mcc = matthews_corrcoef(self.y_val, predictions)

        if val_mcc > self.best_mcc:
            self.best_mcc = val_mcc
            self.model.save_weights(self.checkpoint_file, overwrite=True, save_format=None, options=None)
        
        self.val_f1s.append(val_f1)
        self.val_mccs.append(val_mcc)
        self.times.append(time.time() - self.start_time)
        self.patience_buffer = self.patience_buffer[1:]
        self.patience_buffer.append(val_mcc)

        if self.verbose > 0:
            print("\n- val_f1: %f - val_mcc %f" %(val_f1, val_mcc))

        p_stop = True
        max_mcc = max(self.val_mccs[:-len(self.patience_buffer)], default=0)
        max_buffer = max(self.patience_buffer, default=0)
        if max_mcc + self.min_delta <= max_buffer:
            p_stop = False

        if val_mcc > self.threshold or p_stop:
            self.model.stop_training = True

    def get_metrics(self):
        return self.train_f1s, self.train_mccs, self.val_f1s, self.val_mccs, self.times


parser = argparse.ArgumentParser()
parser.add_argument("-d", help=f"Dataset name {list(DATASETS.keys())}", default="IOT_DNL")
parser.add_argument("-o", help=f"Optimizer {list(OPTIMIZERS.keys())}", default="Adam")
parser.add_argument("-s", help="MCC score to achieve", default=1, type=float)
parser.add_argument("-p", help="Patience", default=5, type=int)
parser.add_argument("-md", help="Minimum Delta", default=0.001, type=float)
parser.add_argument("-lr", help="Learning rate", default=0.001, type=float)
parser.add_argument("-e", help="Number of epochs", default=100, type=int)
parser.add_argument("-b", help="Batch size", default=1024, type=int)
args = parser.parse_args()

if args.d not in DATASETS.keys():
    raise ValueError(f"Dataset name must be one of {list(DATASETS.keys())}")

if args.o not in OPTIMIZERS.keys():
    raise ValueError(f"Optimizer name must be one of {list(OPTIMIZERS.keys())}")

folder = f"{args.d}/data"
dataset_util = DATASETS[args.d]
x_train, y_train = dataset_util.load_training_data()
x_val, y_val = dataset_util.load_validation_data()
output = pathlib.Path(f"{args.d}/single_training")
output.mkdir(parents=True, exist_ok=True)

print(f"Shape of the train data: {x_train.shape}")
print(f"Shape of the validation data: {x_val.shape}")

# Create model
model = dataset_util.create_model()

# compile the model
model.compile(
    optimizer=OPTIMIZERS[args.o](learning_rate=args.lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


start = time.time()

custom_metric = CustomCallback(x_val, y_val, start, 2, output/"checkpoint_file.keras", args.s, args.p, args.md)

history = model.fit(
    x_train, 
    y_train,
    epochs=args.e,
    batch_size=args.b,
    verbose=1,
    callbacks=[custom_metric],
)

end = time.time()
print(f"Training time: {end - start} seconds")

metrics = custom_metric.get_metrics()

history.history["f1_val"] = metrics[2]
history.history["mcc_val"] = metrics[3]
history.history["times"] = metrics[4]

history = json.dumps(history.history)

f = open(output/"train_history.json", "w")
f.write(history)
f.close()

model.load_weights(output/"checkpoint_file.keras")
model.save(output/"single_training.keras")


for type_, x_, y_ in (
    ("train", x_train, y_train),
    ("validation", x_val, y_val),
):
    print(f"\n\n{type_} results")
    print(f"Number of samples: {x_.shape[0]}")
    y_pred = model.predict(x_)
    y_pred = np.argmax(y_pred, axis=1)
    print(f"Confusion matrix:\n{confusion_matrix(y_, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_, y_pred)}")
    print(f"F1 score: {f1_score(y_, y_pred, average='macro')}")
    print(f"MCC: {matthews_corrcoef(y_, y_pred)}")

print(f"\n\nTraining time: {end - start} seconds")
