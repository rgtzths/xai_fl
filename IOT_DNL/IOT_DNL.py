from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

from Util import Util

class IOT_DNL(Util):

    def __init__(self):
        super().__init__("IOT_DNL")



    def data_processing(self):
        dataset = f"{self.name}/data/data.csv"
        output = f"{self.name}/data"
        Path(output).mkdir(parents=True, exist_ok=True)

        data = pd.read_csv(dataset)
        data.dropna()
        X = data.drop('normality', axis=1)
        X = X.drop('frame.number', axis=1)
        X = X.drop('frame.time', axis=1)
        y = data['normality']
        n_samples=X.shape[0]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        scaler = StandardScaler()
        x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
        x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, shuffle=True)

        print(f"\nTotal samples {n_samples}")
        print(f"Shape of the train data: {x_train.shape}")
        print(f"Shape of the validation data: {x_val.shape}")
        print(f"Shape of the test data: {x_test.shape}\n")

        # Save the data
        x_train.to_csv(f"{output}/X_train.csv", index=False)
        x_val.to_csv(f"{output}/X_val.csv", index=False)
        x_test.to_csv(f"{output}/X_test.csv", index=False)
        y_train.to_csv(f"{output}/y_train.csv", index=False)
        y_val.to_csv(f"{output}/y_val.csv", index=False)
        y_test.to_csv(f"{output}/y_test.csv", index=False)



    def data_division(self, n_workers):
        return super().data_division(n_workers)
    


    def load_training_data(self):
        return super().load_training_data()
    


    def load_validation_data(self):
        return super().load_validation_data()
    


    def load_test_data(self):
        return super().load_test_data()
    


    def load_worker_data(self, n_workers, worker_id):
        return super().load_worker_data(n_workers, worker_id)
    


    def create_model(self):
        # Optimizer: Adam
        # Learning rate: 0.000005
        return tf.keras.models.Sequential([
            # flatten layer
            tf.keras.layers.Flatten(input_shape=(11,)),
            # hidden layers
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation='relu'),
            # output layer
            tf.keras.layers.Dense(6, activation='softmax')
        ])