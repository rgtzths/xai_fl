#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Rafael Teixeira'
__version__ = '0.1'
__email__ = 'rafaelgteixeira@ua.pt'
__status__ = 'Development'

import pandas as pd
import numpy as np
import pathlib
import random
from imblearn.under_sampling import RandomUnderSampler
import gc
import psutil


def divide_dataset(dataset_path, n_divisions, output_path, seed, sample_per, train_clusters):
    random.seed(seed)
    rus = RandomUnderSampler(random_state=seed, sampling_strategy=sample_per)
    identifiers = ["site_id", "mlid", "datetime"]

    labels = ["rlf", "1-day-predict", "5-day-predict"]

    output = pathlib.Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    dataset =  pathlib.Path(dataset_path)

    clusters = [x for x in dataset.iterdir() if x.is_dir()]

    test_clusters = len(clusters) - train_clusters

    random.shuffle(clusters)

    subset_size = train_clusters//n_divisions
    path_idx = 0

    for i in range(1, n_divisions+1):
        final_X_array = None
        final_y_array = None
        x_subset_file = open(output/f"x_train_subset_{i}.csv", "wb")
        y_subset_file = open(output/f"y_train_subset_{i}.csv", "wb")
        print(f"Combining subset {i}'s files")
        for j in range(subset_size):
            cluster = clusters[path_idx]

            for folder in [x for x in cluster.iterdir() if x.is_dir()]:
                for file in [x for x in folder.iterdir() if x.is_file()]:
                    df = pd.read_csv(file)

                    time_sentitive_features = [feature for feature in df.columns if feature not in labels and feature not in identifiers]

                    if final_X_array is None:
                        final_X_array = df[time_sentitive_features].values
                        final_y_array = df[["5-day-predict"]].values
                    else:
                        final_X_array = np.concatenate( (final_X_array, df[time_sentitive_features].values), axis=0)
                        final_y_array = np.concatenate( (final_y_array, df[["5-day-predict"]].values), axis=0)

            if psutil.virtual_memory().available < psutil.virtual_memory().total*0.5:
                print("Undersampling")
                final_y_array = final_y_array.astype('int') 
                final_X_array, final_y_array = rus.fit_resample(final_X_array, final_y_array)

                print("Saving")
                np.savetxt(x_subset_file, final_X_array, delimiter=",", fmt="%5.2f")
                np.savetxt(y_subset_file, final_y_array, delimiter=",", fmt="%d")
                del final_X_array
                del final_y_array
                gc.collect()

                final_X_array = None
                final_y_array = None
                        

            path_idx += 1
        print("Undersampling")
        final_y_array = final_y_array.astype('int') 
        final_X_array, final_y_array = rus.fit_resample(final_X_array, final_y_array)

        print("Saving")
        np.savetxt(x_subset_file, final_X_array, delimiter=",", fmt="%5.2f")
        np.savetxt(y_subset_file, final_y_array, delimiter=",", fmt="%d")
        del final_X_array
        del final_y_array
        gc.collect()

        x_subset_file.close()
        y_subset_file.close()

    final_X_cv = None
    final_y_cv = None
    x_cv_file = open(output/"x_cv.csv", "wb")
    y_cv_file = open(output/"y_cv.csv", "wb")

    for j in range(test_clusters):
        cluster = clusters[path_idx]

        for folder in [x for x in cluster.iterdir() if x.is_dir()]:
            for file in [x for x in folder.iterdir() if x.is_file()]:
                df = pd.read_csv(file)

                time_sentitive_features = [feature for feature in df.columns if feature not in labels and feature not in identifiers]

                if final_X_cv is None:
                    final_X_cv = df[time_sentitive_features].values
                    final_y_cv = df[["5-day-predict"]].values
                else:
                    final_X_cv = np.concatenate( (final_X_cv, df[time_sentitive_features].values), axis=0)
                    final_y_cv = np.concatenate( (final_y_cv, df[["5-day-predict"]].values), axis=0)
        
        if psutil.virtual_memory().available < psutil.virtual_memory().total*0.5:

            print("Saving")
            np.savetxt(x_cv_file, final_X_array, delimiter=",", fmt="%5.2f")
            np.savetxt(y_cv_file, final_y_array, delimiter=",", fmt="%d")
            del final_X_cv
            del final_y_cv
            gc.collect()

            final_X_cv = None
            final_y_cv = None
        path_idx += 1

    np.savetxt(output/f"x_cv.csv", final_X_cv, delimiter=",", fmt="%10.2f")
    np.savetxt(output/f"y_cv.csv", final_y_cv, delimiter=",", fmt="%d")
    x_cv_file.close()
    y_cv_file.close()

