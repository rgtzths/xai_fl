from pathlib import Path
import pandas as pd


class Util:

    def __init__(self, name):
        """
        Initialize the class
        """
        self.name = name



    def data_processing(self):
        """
        Process the raw data
        Save the processed data in the data folder as
        X_train.csv, X_val.csv, X_test.csv, y_train.csv, y_val.csv, y_test.csv
        to use default methods.
        """
        pass



    def data_division(self, n_workers):
        """
        Divide the data into n_workers subsets
        Save the subsets in [n_workers]_workers folder as
        x_train_subset_[i].csv, y_train_subset_[i].csv, where i = 1, 2, ..., n_workers
        """
        print(f'Dividing the data for {n_workers} workers...')
        x_train_path = f"{self.name}/data/X_train.csv"
        y_train_path = f"{self.name}/data/y_train.csv"
        output_path = f"{self.name}/data/{n_workers}_workers"
        output = Path(output_path)
        output.mkdir(parents=True, exist_ok=True)
        x_train = pd.read_csv(x_train_path)
        y_train = pd.read_csv(y_train_path)
        n = x_train.shape[0]
        subset_size = n // n_workers
        file_number = 1
        for i in range(0, subset_size*n_workers, subset_size):
            print(f'Creating subset {file_number}...')
            x_train_subset = x_train.iloc[i:i+subset_size]
            y_train_subset = y_train.iloc[i:i+subset_size]
            x_train_subset.to_csv(f'{output_path}/x_train_subset_{file_number}.csv', index=False, header=None)
            y_train_subset.to_csv(f'{output_path}/y_train_subset_{file_number}.csv', index=False, header=None)
            file_number += 1



    def load_training_data(self):
        """
        Load the training data
        """
        x_train = pd.read_csv(f"{self.name}/data/X_train.csv")
        y_train = pd.read_csv(f"{self.name}/data/y_train.csv")
        return x_train, y_train



    def load_validation_data(self):
        """
        Load the validation data
        """
        x_val = pd.read_csv(f"{self.name}/data/X_val.csv")
        y_val = pd.read_csv(f"{self.name}/data/y_val.csv")
        return x_val, y_val



    def load_test_data(self):
        """
        Load the test data
        """
        x_test = pd.read_csv(f"{self.name}/data/X_test.csv")
        y_test = pd.read_csv(f"{self.name}/data/y_test.csv")
        return x_test, y_test



    def load_worker_data(self, n_workers, worker_id):
        """
        Load the data of a worker
        """
        x_train = pd.read_csv(f"{self.name}/data/{n_workers}_workers/x_train_subset_{worker_id}.csv", header=None)
        y_train = pd.read_csv(f"{self.name}/data/{n_workers}_workers/y_train_subset_{worker_id}.csv", header=None)
        return x_train, y_train



    def create_model(self):
        """
        Create the model
        """
        pass
