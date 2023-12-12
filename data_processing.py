import argparse

from config import DATASETS

parser = argparse.ArgumentParser()
parser.add_argument("-d", help=f"Dataset name {list(DATASETS.keys())}", default="IOT_DNL")
args = parser.parse_args()

if args.d not in DATASETS.keys():
    raise ValueError(f"Dataset name must be one of {list(DATASETS.keys())}")

dataset_util = DATASETS[args.d]
dataset_util.data_processing()
