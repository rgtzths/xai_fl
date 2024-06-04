import argparse
import tensorflow as tf
from pathlib import Path
from sklearn.utils import resample
import numpy as np

from config import XAI_IMAGES, DATASETS

parser = argparse.ArgumentParser()
parser.add_argument("-d", help=f"Dataset name {list(DATASETS.keys())}", default="MNIST")
parser.add_argument("-x", help="XAI method to use", default="gradCAM")
parser.add_argument("-p", help="Percentage of the test dataset to use", default=10, type=float)
args = parser.parse_args()

if args.x not in XAI_IMAGES:
    raise ValueError(f"XAI method {args.x} not found")

# all .keras and .h5 files in models folder
models = Path(f"{args.d}/models").glob("*.keras")
models = list(models) + list(Path(f"{args.d}/models").glob("*.h5"))
models = [str(model) for model in models]
print(f"Found {len(models)} models")
print(f"Models: {models}")

folder = Path(f"{args.d}/xai/{args.x}")
folder.mkdir(parents=True, exist_ok=True)
# get all files in the folder
files = Path(folder).glob("*")
# remove the extension and get only the name
files = [str(file).split("/")[-1].split(".")[0] for file in files]
# remove from models the ones that already have results
models = [model for model in models if model.split("/")[-1].split(".")[0]]
print(f"Models to process: {len(models)}")
print(f"Models: {models}")

dataset_util = DATASETS[args.d]
x_train, y_train = dataset_util.load_training_data()
x_test, y_test = dataset_util.load_validation_data()
n_samples = int(x_test.shape[0] * args.p / 100)
x_test, y_test = resample(x_test, y_test, n_samples=n_samples, random_state=42, stratify=y_test)
print(f"X_test shape: {x_test.shape}")

for model_path in models:
    print(f"Model: {model_path.split('/')[-1]}")
    model = tf.keras.models.load_model(model_path)
    model_name = model_path.split("/")[-1].split(".")[0]

    sample = x_test[0]
    explainer = XAI_IMAGES[args.x](model, sample.shape)

    # get one heatmap to get its shape
    heatmap = explainer.heatmap(sample)
    explanations = np.empty((x_test.shape[0], heatmap.shape[0], heatmap.shape[1]))

    for i, img in enumerate(x_test):
        explanations[i] = explainer.heatmap(img)

    with open(folder / f"{model_name}.npy", "wb") as f:
        np.save(f, explanations)
