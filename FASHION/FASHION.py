import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
import numpy as np

from Util import Util

class FASHION(Util):

    def __init__(self):
        super().__init__("FASHION")


    def data_processing(self):
        output = f"{self.name}/data"

        Path(output).mkdir(parents=True, exist_ok=True)
        
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        n_samples = x.shape[0]

        # normalize the data
        x = x.astype('float32')
        x = x / 255.0

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, shuffle=True)

        print(f"\nTotal samples {n_samples}")
        print(f"Shape of the train data: {x_train.shape}")
        print(f"Shape of the validation data: {x_val.shape}")
        print(f"Shape of the test data: {x_test.shape}\n")

        np.save(f"{output}/X_train.npy", x_train)
        np.save(f"{output}/X_val.npy", x_val)
        np.save(f"{output}/X_test.npy", x_test)
        np.save(f"{output}/y_train.npy", y_train)
        np.save(f"{output}/y_val.npy", y_val)
        np.save(f"{output}/y_test.npy", y_test)

    
    def data_division(self, n_workers):
        output = f"{self.name}/data/{n_workers}_workers"
        Path(output).mkdir(parents=True, exist_ok=True)

        x_train = np.load(f"{self.name}/data/X_train.npy")
        y_train = np.load(f"{self.name}/data/y_train.npy")
        n = x_train.shape[0]
        subset_size = n // n_workers
        file_number = 1
        for i in range(0, subset_size*n_workers, subset_size):
            print(f'Creating subset {file_number}...')
            x_train_subset = x_train[i:i+subset_size]
            y_train_subset = y_train[i:i+subset_size]
            np.save(f'{output}/x_train_subset_{file_number}.npy', x_train_subset)
            np.save(f'{output}/y_train_subset_{file_number}.npy', y_train_subset)
            file_number += 1


    def load_training_data(self):
        x_train = np.load(f"{self.name}/data/X_train.npy")
        y_train = np.load(f"{self.name}/data/y_train.npy")
        return x_train, y_train
    

    def load_validation_data(self):
        x_val = np.load(f"{self.name}/data/X_val.npy")
        y_val = np.load(f"{self.name}/data/y_val.npy")
        return x_val, y_val
    

    def load_test_data(self):
        x_test = np.load(f"{self.name}/data/X_test.npy")
        y_test = np.load(f"{self.name}/data/y_test.npy")
        return x_test, y_test
    

    def load_worker_data(self, n_workers, worker_id):
        x_train = np.load(f"{self.name}/data/{n_workers}_workers/x_train_subset_{worker_id}.npy")
        y_train = np.load(f"{self.name}/data/{n_workers}_workers/y_train_subset_{worker_id}.npy")
        return x_train, y_train


    def create_model(self):
        return tf.keras.models.Sequential([
            # input layer
            # hidden layers
            # output layer
        ])