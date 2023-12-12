import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import glob
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

CORR = {
    "Pearson": pearsonr,
    "Spearman": spearmanr
}

parser = argparse.ArgumentParser()
parser.add_argument("-d", help="Dataset to use", default="IOT_DNL")
parser.add_argument("-x", help="XAI method to use", default="fanova")
args = parser.parse_args()

# check if folder exists
if not Path(args.d).exists():
    print(f"Folder {args.d} not found")
    exit(1)

# get folders inside xai folder
xai_methods = Path(f"{args.d}/xai").glob("*")
xai_methods = [str(method) for method in xai_methods if method.is_dir()]
xai_methods = [method.split("/")[-1] for method in xai_methods]
if args.x not in xai_methods:
    print(f"XAI method {args.x} not found in {xai_methods}")
    exit(1)

xai_nn = pd.read_csv(f"{args.d}/xai/{args.x}/single_training.csv")
all_files = glob.glob(f"{args.d}/xai/{args.x}/*.csv")
all_files.remove(f"{args.d}/xai/{args.x}/single_training.csv")
print("Found {} files".format(len(all_files)))
print("Files: {}".format(all_files))

# read all files into a list of dataframes
data = []
for filename in all_files:
    df = pd.read_csv(filename)
    data.append(df)


features = xai_nn['feature']
colls = xai_nn.columns[1:]

for corr_type, corr_fn in CORR.items():
    correlations = []
    for col in colls:
        corr = []
        for df in data:
            corr.append(corr_fn(xai_nn[col], df[col])[0])
        correlations.append(corr)

    plt.figure(figsize=(10, 6))
    plt.title(f"{corr_type} Correlation to single training")
    plt.xlabel("Class")
    plt.ylabel("Correlation Coefficient")
    for i, file_ in enumerate(all_files):
        plt.bar(np.arange(len(colls)) + i * 0.2, [corr[i] for corr in correlations], width=0.2, label=f"{file_.split('/')[-1].split('.')[0]}")
    plt.xticks(np.arange(len(colls)) + (len(all_files) - 1) * 0.2 / 2, colls)
    plt.legend(loc='lower right')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{args.d}/xai/{args.x}/correlation_{corr_type}.png")

