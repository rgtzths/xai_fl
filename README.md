# XAI-FL

## Structure

This repository has the following structure:
```
├── FL/
├── XAI/
├── [DATASET]/
├── config.py
├── correlation.py
├── data_division.py
├── feature_analysis.py
├── federated_learning.py
├── model_eval.py
├── single_training.py
├── plot_history.py
├── Util.py
└── xai.py
```

Other folders contain data and results for specific datasets, which have the following structure:
```
├── data/
├── fl/
├── models/
├── xai/
└── [DATASET].py
```

For the main structure:
- FL/ contains the implementation of the federated learning algorithms, which are used in federated_learning.py. 
- XAI/ contains the implementation of the XAI algorithms, which are used in xai.py. 
- config.py defines the datasets and xai methods to be used in the experiments.
- Util.py contains a class which all datasets must inherit from.
- The other files are used to run experiments, use --help to see the options.

For the dataset structure:
- data/ contains the data files, including the train, test, validation and specific workers' data.
- fl/ contains the results of the federated learning algorithms, including the models and the training logs, for each experiment.
- models/ contains the models to be used in XAI algorithms.
- xai/ contains the results for each XAI algorithm, including the .csv files with the explanations and the plots.
- [DATASET].py contains the implementation of the dataset, including the data loading and the data division.

## Setup

First, create a virtual environment and install the requirements:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, we need to fix an issue in fanova, which is not compatible with latest versions of Numpy, and to do so we need to change the line 286 in venv/lib/python3.11/site-packages/fanova/fanova.py from:
```
            sample = np.full(self.n_dims, np.nan, dtype=np.float)
```
to:
```
            sample = np.full(self.n_dims, np.nan, dtype=np.float64)
```

Download the datasets and put them in the data/ folder of each dataset (links found in the data/README.md files of each dataset).

Finally, run the data_processing.py and data_division.py files for each dataset, to process the data and divide it into train, test, validation and workers' data.

## Running the experiments

- Run single_training.py for each dataset
- Run federated_learning.py for the experiments you want to run
- Copy the models for evaluation to the models/ folder of each dataset
- Run xai.py for the experiments you want to run
- Run correlation.py and/or feature_analysis.py for a given dataset and xai method to save the results in the corresponding folder

## Results

### Network Slicing

#### SHAP

##### PCC

![shap correlation values using PCC](Slicing5G/xai/shap/correlation_Pearson.png)

##### SRCC

![shap correlation values using SRCC](Slicing5G/xai/shap/correlation_Spearman.png)


#### fANOVA 

##### PCC

![fanova correlation values using PCC](Slicing5G/xai/fanova/correlation_Pearson.png)

##### SRCC

![fanova correlation values using SRCC](Slicing5G/xai/fanova/correlation_Spearman.png)


### Intrusion Detection

#### SHAP

##### PCC

![shap correlation values using PCC](IOT_DNL/xai/shap/correlation_Pearson.png)

##### SRCC

![shap correlation values using SRCC](IOT_DNL/xai/shap/correlation_Spearman.png)


#### fANOVA 

##### PCC

![fanova correlation values using PCC](IOT_DNL/xai/fanova/correlation_Pearson.png)

##### SRCC

![fanova correlation values using SRCC](IOT_DNL/xai/fanova/correlation_Spearman.png)


## Authors

* **Rafael Teixeira** - [rgtzths](https://github.com/rgtzths)
* **Leonardo Almeida** - [leoalmPT](https://github.com/leoalmPT/)
* **Pedro Rodrigues** - [pedro535](https://github.com/pedro535/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details