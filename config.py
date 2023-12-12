import tensorflow as tf

from IOT_DNL.IOT_DNL import IOT_DNL
from Slicing5G.Slicing5G import Slicing5G
from XAI.xai_fanova import results as results_fanova
from XAI.xai_shap import results as results_shap

DATASETS = {
    "IOT_DNL": IOT_DNL(),
    "Slicing5G": Slicing5G(),
}

OPTIMIZERS = {
    "SGD": tf.keras.optimizers.SGD,
    "Adam": tf.keras.optimizers.Adam
}

XAI = {
    "fanova": results_fanova,
    "shap": results_shap
}
