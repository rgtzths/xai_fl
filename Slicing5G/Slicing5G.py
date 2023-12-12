from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

from Util import Util

class Slicing5G(Util):

    def __init__(self):
        super().__init__("Slicing5G")



    def data_processing(self):
        dataset = f"{self.name}/data/raw_data.xlsx"
        output = f"{self.name}/data"
        # create the output folder if it does not exist
        Path(output).mkdir(parents=True, exist_ok=True)

        df = pd.read_excel(dataset, sheet_name="Model_Inputs_Outputs")
        le = LabelEncoder()
        del df["Unnamed: 0"]
        #Transform features into categories
        df["Use CaseType (Input 1)"] = le.fit_transform(df["Use CaseType (Input 1)"])
        df["LTE/5G UE Category (Input 2)"] = df["LTE/5G UE Category (Input 2)"].astype(str)
        df["LTE/5G UE Category (Input 2)"] = le.fit_transform(df["LTE/5G UE Category (Input 2)"])
        df["Technology Supported (Input 3)"] = le.fit_transform(df["Technology Supported (Input 3)"])
        df["Day (Input4)"] = le.fit_transform(df["Day (Input4)"])
        df["QCI (Input 6)"] = le.fit_transform(df["QCI (Input 6)"])
        df["Packet Loss Rate (Reliability)"] = le.fit_transform(df["Packet Loss Rate (Reliability)"])
        df["Packet Delay Budget (Latency)"] = le.fit_transform(df["Packet Delay Budget (Latency)"])
        df["Slice Type (Output)"] = le.fit_transform(df["Slice Type (Output)"])

        X = df.drop('Slice Type (Output)', axis=1)
        y = df['Slice Type (Output)']
        n_samples=X.shape[0]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
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
        # Learning rate: 0.00001
        return tf.keras.models.Sequential([
            # flatten layer
            tf.keras.layers.Flatten(input_shape=(8,)),
            # hidden layers
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            # output layer
            tf.keras.layers.Dense(3, activation="softmax")
        ])