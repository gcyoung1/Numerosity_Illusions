import matplotlib.pyplot as plt
import time
import argparse
import os
import numpy as np
import pandas as pd

from . import utility_functions as utils

if __name__ == '__main__':

    start_time = time.time()

    #Command Line Arguments 
    parser = argparse.ArgumentParser(description='Plot the tuning curves on a particular dataset.')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Name of experiment')
    
    args = parser.parse_args()
    # reconcile arguments
    print('running with args:')
    print(args)

    # Get path to experiment directory
    experiment_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/models', args.experiment_name)

    df = pd.read_csv(os.path.join(experiment_path,"percent_numerosity_neurons.csv"))
    selection_methods = df['selection_method'].unique()

    num_subplots = len(selection_methods)
    fig_side=num_subplots*5
    fig, subplots = plt.subplots(1,num_subplots,figsize=(fig_side,fig_side))
    fig.suptitle(f"Percent Numerosity Neurons",size='xx-large')
    subplots_list = np.ravel(subplots)
    
    for i,selection_method in enumerate(selection_methods):
        selection_df = df[df['selection_method']==selection_method]
        for model_name in selection_df['model_name'].unique():
            model_df = selection_df[selection_df['model_name'] == model_name]
            for layer in model_df['layer'].unique():
                layer_df = model_df[model_df['layer'] == layer].sort_values(by='dataset_name', ascending=True)
                subplots_list[i].plot([10,30,100],layer_df['percent_of_layer'], label=f"{model_name}_{layer}")
                subplots_list[i].set_title(selection_method)
                subplots_list[i].set_ylim((0,0.25))

    subplots_list[0].legend()
    fig.savefig(os.path.join(experiment_path, "percent_numerosity_neurons.png"))

    print('Total Run Time:')
    print("--- %s seconds ---" % (time.time() - start_time))
