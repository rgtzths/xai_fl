import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-d", help="Dataset to use", default="IOT_DNL")
parser.add_argument("-x", help="XAI method to use", default="fanova")
args = parser.parse_args()

# check if folder exists
if not Path(args.d).exists():
    print(f"Folder {args.d} not found")
    exit(1)

all_files = glob.glob(f"{args.d}/xai/{args.x}/*.csv")
print("Found {} files".format(len(all_files)))
print("Files: {}".format(all_files))

# read all files into a list of dataframes
data = []
for filename in all_files:
    df = pd.read_csv(filename)
    data.append(df)

features = data[0]['feature'] 
num_files = len(all_files)
colls = data[0].columns
for column in colls[1:]:
    print(column)
    plt.figure(figsize=(10, 6))
    plt.title(column.capitalize() + " Values for Each Feature")
    plt.xlabel("Feature")
    plt.ylabel("Importance (%)")

    for i in range(num_files):
        plt.bar(np.arange(len(features)) + i * 0.2, data[i][column], width=0.2, label=f"{all_files[i].split('/')[-1].split('.')[0]}")

    plt.xticks(np.arange(len(features)) + (num_files - 1) * 0.2 / 2, features)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{args.d}/xai/{args.x}/{column.capitalize()}.png")

