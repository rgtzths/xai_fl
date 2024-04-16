import matplotlib.pyplot as plt
import argparse
import glob
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import json

CORR = {
    "Pearson": pearsonr,
    "Spearman": spearmanr
}

parser = argparse.ArgumentParser()
parser.add_argument("-d", help="Dataset to use", default="MNIST")
parser.add_argument("-x", help="XAI method to use", default="gradCAM")
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

all_files = glob.glob(f"{args.d}/xai/{args.x}/*.json")
all_files.remove(f"{args.d}/xai/{args.x}/single_training.json")
print("Found {} files".format(len(all_files)))
print("Files: {}".format(all_files))

# read single training explanations
with open(f"{args.d}/xai/{args.x}/single_training.json", 'r') as f:
    xai_nn = json.load(f).values()
    xai_nn = list(xai_nn)

# read all explanations
explanations = []
for filename in all_files:
    with open(filename, 'r') as f:
        exp = json.load(f).values()
        explanations.append(list(exp))


for corr_type, corr_fn in CORR.items():
    correlations = [corr_fn(xai_nn, exp)[0] for exp in explanations]

    plt.figure(figsize=(10, 6))
    plt.title(f"{corr_type} Correlation to single training")
    plt.xlabel("FL type")
    plt.ylabel("Correlation Coefficient")
    
    fl_types = [file_.split('/')[-1].split('.')[0] for file_ in all_files]
    plt.bar(fl_types, correlations)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{args.d}/xai/{args.x}/correlation_{corr_type}.png")
