import tensorflow as tf

from IOT_DNL.IOT_DNL import IOT_DNL
from Slicing5G.Slicing5G import Slicing5G
from MNIST.MNIST import MNIST
from CIFAR.CIFAR import CIFAR
from FASHION.FASHION import FASHION
from UNSW.UNSW import UNSW

# from XAI.xai_fanova import results as results_fanova
from XAI.xai_shap import results as results_shap
from XAI.PI import results as results_pi
from XAI.PDV import results as results_pdv
from XAI.GradCAM import GradCAM
from XAI.IntegratedGrads import IntegratedGrads

DATASETS = {
    "IOT_DNL": IOT_DNL(),
    "Slicing5G": Slicing5G(),
    "MNIST": MNIST(),
    "CIFAR": CIFAR(),
    "FASHION": FASHION(),
    "UNSW": UNSW()
}

OPTIMIZERS = {
    "SGD": tf.keras.optimizers.SGD,
    "Adam": tf.keras.optimizers.Adam
}

XAI = {
    # "fanova": results_fanova,
    "shap": results_shap,
    "PI": results_pi,
    "PDV": results_pdv
}

XAI_IMAGES = {
    "gradCAM": GradCAM,
    "integratedGrads": IntegratedGrads
}
