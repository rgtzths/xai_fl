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
    def __init__(self, X_val,  y_val, start_time, verbose=0, checkpoint_file="model_checkpoint.keras", threshold=1):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val

        self.start_time = start_time
        self.verbose = verbose
        self.best_mcc = -1
        self.checkpoint_file = checkpoint_file
        self.threshold = threshold

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

        if self.verbose > 0:
            print("\n- val_f1: %f - val_mcc %f" %(val_f1, val_mcc))

        if val_mcc > self.threshold:
            self.model.stop_training = True

    def get_metrics(self):
        return self.train_f1s, self.train_mccs, self.val_f1s, self.val_mccs, self.times


parser = argparse.ArgumentParser()
parser.add_argument("-d", help=f"Dataset name {list(DATASETS.keys())}", default="IOT_DNL")
parser.add_argument("-o", help=f"Optimizer {list(OPTIMIZERS.keys())}", default="Adam")
parser.add_argument("-s", help="MCC score to achieve", default=1, type=float)
parser.add_argument("-lr", help="Learning rate", default=0.001, type=float)
parser.add_argument("-e", help="Number of epochs", default=200, type=int)
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

custom_metric = CustomCallback(x_val, y_val, start, 2, output/"checkpoint_file.keras", args.s)

history = model.fit(
    x_train, 
    y_train,
    epochs=args.e,
    batch_size=args.b,
    verbose=1,
    callbacks=[custom_metric],
)

end = time.time()

metrics = custom_metric.get_metrics()

history.history["f1_val"] = metrics[2]
history.history["mcc_val"] = metrics[3]
history.history["times"] = metrics[4]

history = json.dumps(history.history)

f = open(output/"train_history.json", "w")
f.write(history)
f.close()

model.load_weights(output/"checkpoint_file.keras")


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

model.save(f"{args.d}/models/single_training.keras")

print(f"\n\nTraining time: {end - start} seconds")