import argparse
import pandas as pd
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, matthews_corrcoef

from config import DATASETS

parser = argparse.ArgumentParser()
parser.add_argument("-d", help=f"Dataset name {list(DATASETS.keys())}", default="IOT_DNL")
args = parser.parse_args()

model_traslation = {
    "single_training": "Single Host Training",
    "centralized_sync" : "Centralized Sync",
    "centralized_async": "Centralized Async",
    "decentralized_sync":"Decentralized Sync",
    "decentralized_async":"Decentralized Async"
}

if args.d not in DATASETS.keys():
    raise ValueError(f"Dataset name must be one of {list(DATASETS.keys())}")

models = Path(f"{args.d}/models").glob("*.keras")
models = list(models) + list(Path(f"{args.d}/models").glob("*.h5"))
models = [str(model) for model in models]
print(f"Found {len(models)} models")
print(f"Models: {models}")

df = pd.DataFrame(columns=["model", "train_acc", "train_f1", "train_mcc", "val_acc", "val_f1", "val_mcc", "test_acc", "test_f1", "test_mcc"])

dataset_util = DATASETS[args.d]
x_train, y_train = dataset_util.load_training_data()
x_val, y_val = dataset_util.load_validation_data()
x_test, y_test = dataset_util.load_test_data()

#print(f"\n\nShape of the train data: {x_train.shape}")
#print(f"Shape of the validation data: {x_val.shape}")
#print(f"Shape of the test data: {x_test.shape}")
print("\\multicolumn{7}{c}{\\textbf{"+args.d+"}}\\\\")
for model_name in model_traslation:
    model = None
    for model_path in models:
        if model_name == model_path.split("/")[-1].split(".")[0]:
            model = tf.keras.models.load_model(model_path)
            break
    if model is None:
        print(f"{model_traslation[model_name]} & - & - & - & - & - & - \\\\")
        res = [model_traslation[model_name]]
        res.append(0)
        res.append(0)
        res.append(0)
    else:
        text = f"{model_traslation[model_name]}"
        res = [model_traslation[model_name]]
        for type_, x_, y_ in (
            ("train", x_train, y_train),
            ("validation", x_val, y_val),
            ("test", x_test, y_test)
        ):
            #print(f"\n\n{type_} results")
            #print(f"Number of samples: {x_.shape[0]}")
            y_pred = model.predict(x_, verbose=0)
            y_pred = np.argmax(y_pred, axis=1)
            #print(f"Confusion matrix:\n{confusion_matrix(y_, y_pred)}")
            #print(f"Accuracy: {accuracy_score(y_, y_pred)}")
            #print(f"F1 score: {f1_score(y_, y_pred, average='macro')}")
            #print(f"MCC: {matthews_corrcoef(y_, y_pred)}")
            if type_ != "train":
                text += f"& {accuracy_score(y_, y_pred):.3f} & {f1_score(y_, y_pred, average='macro'):.3f} & {matthews_corrcoef(y_, y_pred):.3f}"

            res.append(accuracy_score(y_, y_pred))
            res.append(f1_score(y_, y_pred, average='macro'))
            res.append(matthews_corrcoef(y_, y_pred))
        print(text+"\\\\")
        
        df.loc[len(df)] = res

df.to_csv(f"{args.d}/models/scores.csv", index=False)




