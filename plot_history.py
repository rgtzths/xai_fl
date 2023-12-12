#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Rafael Teixeira'
__version__ = '0.1'
__email__ = 'rafaelgteixeira@ua.pt'
__status__ = 'Development'

import argparse
import json
import pathlib

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from functools import partial


def plot_model_history(input_folder, file_name):

    plt.rcParams.update({'font.size': 22})
    output = pathlib.Path(input_folder)

    model_history = json.load(open(output/ file_name))

    for key in model_history:
        if key != "times":
            state = 0
            points = []
            values = list(np.arange(0.50,1,0.05))
            for i in range(len(model_history[key])):
                if model_history[key][i] >= values[state]:
                    while state < len(values) and model_history[key][i] >= values[state]:
                        state += 1
                    try:
                        points.append((i+1, model_history[key][i], model_history["times"]["global_times"][i]))
                    except:
                        points.append((i+1, model_history[key][i], model_history["times"][i]))

                    if state == len(values):
                        break

            n_epochs = len(model_history[key])

            fig = plt.figure(figsize=(12, 8))

            plt.plot(range(1, n_epochs+1), model_history[key], label="Validation " +key)
            for i in range(len(points)):
                x = points[i][0]
                y = points[i][1]
                plt.plot(x, y, 'bo')
                plt.text(x + 2.5,  y-0.02, "%.2f" % (points[i][2] / 60), fontsize=18)
                if key == "mcc_val" or key == "mcc":
                    print("%6.2f %6.2f" %(y, points[i][2] / 60))

            plt.xlabel("Nº Epochs")
            plt.ylabel(key)
            plt.title("Evolution of the "+key+" of the model")
            plt.legend()

            fig.savefig(output /(key+".pdf"), dpi=300.0, bbox_inches='tight', format="pdf", orientation="landscape")

    plt.close('all')

def get_average_times(folder, centralized, n_batches):
    output = pathlib.Path(folder)

    average_results = {
        "train" : [],
        "comm_send" : [],
        "comm_recv" : [],
        "conv_send" : [],
        "conv_recv" : [],
        "epochs" : []
    }

    for file in output.glob("worker*"):
        model_history = json.load(open(file))
        for key in average_results:
            if centralized and key != "epochs":
                    temp_array = []
                    for i in range(0, len(model_history["times"][key]), n_batches):
                        temp_array.append(sum(model_history["times"][key][i:i+n_batches]))
                    average_results[key].append(sum(temp_array)/len(temp_array))
                    
            else:
                average_results[key].append(sum(model_history["times"][key])/len(model_history["times"][key]))

 
    # create data
    x = ['train_divided', 'global_training']
    bars = [np.log10(sum(average_results[key])/len(average_results[key]) )for key in average_results if key != "epochs" ]
    global_bar = [0] * len(bars)
    global_bar[0] = np.log10(sum(average_results["epochs"])/len(average_results["epochs"]))
    
    df = pd.DataFrame([['Train_divided'] + bars, ['Global training'] + global_bar],
                  columns=['Analysis', 'Train', 'conv_send', 'comm_send', 'comm_recv', 'conv_recv'])
    
    df.plot(x='Analysis', kind='bar', stacked=True)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test the model')
    parser.add_argument('-f', type=str, help='Input/output folder', default='results/')
    parser.add_argument('-s', type=str, help='Server file name', default='server.json')
    parser.add_argument('-c', type=bool, help='Used centralized or not', default=True)
    parser.add_argument('-n', type=bool, help='Number of batches', default=8)


    args = parser.parse_args()

    #plot_model_history(args.f, args.s)
    get_average_times(args.f, args.c, args.n)