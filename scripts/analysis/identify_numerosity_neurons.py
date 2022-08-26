import os
import time
import argparse
import pickle       
import numpy as np
import pandas as pd
import pingouin as pg
import random

from . import utility_functions as utils
from ..plotting import utility_functions as plotting_utils

def getAnovaDict(df,num_neurons,parameters_header):
    anova_dict = {}
    num_nonzero_entries = df.astype(bool).sum(axis=0)

    # Don't do anova on num_lines if it's always 0
    if num_nonzero_entries['num_lines'] == 0:
        parameters_header.remove('num_lines')

    for i in range(num_neurons):
        # Exclude from contention neurons with 0 activation for all stimuli
        if num_nonzero_entries[f"n{i}"] > 0:
            print(f"n{i}")
            start_time = time.time()
            aov = pg.anova(dv=f'n{i}', between=parameters_header, data=df,detailed=True)
            print(f"Anova took {time.time()-start_time} seconds, total will take {(num_neurons-i)*(time.time()-start_time)/60} more minutes")
            
            # Add to dict 
            anova_dict[f'n{i}'] = {}
            for row in range(len(parameters_header)):
                anova_dict[f'n{i}'][f'{parameters_header[row]}'] = {}
                anova_dict[f'n{i}'][f'{parameters_header[row]}']['np2'] = aov.at[row,'np2']
                anova_dict[f'n{i}'][f'{parameters_header[row]}']['p-unc'] = aov.at[row,'p-unc']

    return anova_dict

def getVarianceDict(df,num_neurons,parameters_header):
    variance_dict = {}
    num_nonzero_entries = df.astype(bool).sum(axis=0)

    # Don't do variance calculation on num_lines if it's always 0
    if num_nonzero_entries['num_lines'] == 0:
        parameters_header.remove('num_lines')

    for i in range(num_neurons):
        # Exclude from contention neurons with 0 activation for all stimuli
        if num_nonzero_entries[f"n{i}"] > 0:
            print(f"n{i}")
            
            start_time = time.time()

            # Add to dict 
            variance_dict[f'n{i}'] = {}
            for row in range(len(parameters_header)):
                variance_dict[f'n{i}'][f'{parameters_header[row]}'] = {}
                variance_dict[f'n{i}'][f'{parameters_header[row]}']['np2'] = aov.at[row,'np2']
                variance_dict[f'n{i}'][f'{parameters_header[row]}']['p-unc'] = aov.at[row,'p-unc']

            print(f"Variance took {time.time()-start_time} seconds, total will take {(num_neurons-i)*(time.time()-start_time)/60} more minutes")
    return variance_dict


def getNumerosityNeurons(anova_dict,parameters_header,selection_method):
    numerosity_neurons = []

    for neuron_id in anova_dict.keys():
        neuron_dict = anova_dict[neuron_id]

        # Initialize assuming no effects
        numerosity_effects = False
        non_numerosity_effects = False

        if selection_method == 'anova':
            # Effect is p-value
            numerosity_p_value = neuron_dict['numerosity']['p-unc']
            numerosity_effects = (numerosity_p_value < 0.01)
            for parameter in parameters_header[1:]:
                non_numerosity_p_value = neuron_dict[parameter]['p-unc']
                non_numerosity_effect = non_numerosity_p_value < 0.01
                non_numerosity_effects = non_numerosity_effects or non_numerosity_effect

        elif selection_method == 'anova1way':
            # Only check for numerosity effect
            numerosity_p_value = neuron_dict['numerosity']['p-unc']
            numerosity_effects = (numerosity_p_value < 0.01)

        elif selection_method == 'variance':
            # Effect is explained variance
            numerosity_variance = neuron_dict['numerosity']['np2']
            numerosity_effects = (numerosity_variance > 0.1)
            for parameter in parameters_header:
                non_numerosity_variance = neuron_dict[parameter]['np2']
                non_numerosity_effect = non_numerosity_variance > 0.01
                non_numerosity_effects = non_numerosity_effects or non_numerosity_effect

        if numerosity_effects and not non_numerosity_effects:
            numerosity_neurons.append(int(neuron_id[1:]))
            anova_dict[neuron_id][selection_method] = True
        else:
            anova_dict[neuron_id][selection_method] = False

    return numerosity_neurons


def sortNumerosityNeurons(numerosity_neurons,numerosities,average_activations):
    max_activation_numerosity = average_activations.idxmax(axis=0)
    sorted_number_neurons = [[] for _ in range(len(numerosities))]
    for idx in numerosity_neurons:
        num = max_activation_numerosity[f'n{idx}']
        sorted_number_neurons[num].append(idx)
    return sorted_number_neurons

def identifyNumerosityNeurons(dataset_path,selection_method):
    method_path = os.path.join(dataset_path,f'{selection_method}_')
    parameters_header = ['numerosity', 'size','spacing','num_lines']

    df = utils.getActivationDataFrame(dataset_path,'activations')

    if not os.path.exists(os.path.join(dataset_path, 'numerosities.npy')):
        numerosities = df['numerosity'].unique().tolist()
        numerosities.sort()
        np.save(os.path.join(dataset_path, 'numerosities.npy'), numerosities)
    else:
        numerosities = np.load(os.path.join(dataset_path, 'numerosities.npy'))

    if not os.path.exists(os.path.join(dataset_path, 'anova_dict.pkl')):
        print("Performing anovas...")
        num_neurons = len(df.columns)-len(parameters_header)
        anova_dict = getAnovaDict(df,num_neurons,parameters_header)
        with open(os.path.join(dataset_path, 'anova_dict.pkl'), 'wb') as f:
            pickle.dump(anova_dict, f)
        f.close()
    else:
        print("Loading anovas...")
        with open(os.path.join(dataset_path, 'anova_dict.pkl'), 'rb') as f:
            anova_dict = pickle.load(f)
    import pdb;pdb.set_trace()
    if not os.path.exists(method_path + 'numerosityneurons.npy'):
        print("Identifying numerosity neurons...")
        numerosity_neurons = getNumerosityNeurons(anova_dict,parameters_header,selection_method)

        print("Sorting numerosity neurons...")
        average_activations = df.groupby(['numerosity'],as_index=False).mean()
        sorted_numerosity_neurons = sortNumerosityNeurons(numerosity_neurons,numerosities,average_activations)
        np.save(method_path + 'numerosityneurons', np.array(sorted_numerosity_neurons))
    else:
        average_activations = df.groupby(['numerosity'],as_index=False).mean()
        sorted_numerosity_neurons = np.load(method_path + 'numerosityneurons.npy')
        numerosity_neurons = np.ravel(sorted_numerosity_neurons)
    
    print("Calculating tuning curves...")
    tuning_curves, std_errs = utils.getTuningCurves(sorted_numerosity_neurons,numerosities,average_activations)
    np.save(method_path+"tuning_curves", tuning_curves)
    np.save(method_path+"std_errs", std_errs)

    # Plotting

    # Plot tuning curves
    if selection_method == 'variance':
        # Make plot of the variance explained by dimension for each neuron
        print("Plotting variance explained...")
        fig = plotting_utils.plotVarianceExplained(anova_dict, numerosity_neurons)
        fig.savefig(os.path.join(dataset_path, "variance_explained"))
    # Make plot of the number of numerosity neurons sensitive to each numerosity
    print("Plotting numerosity histogram...")
    fig = plotting_utils.saveNumerosityHistogram(sorted_numerosity_neurons,numerosities)
    fig.savefig(os.path.join(dataset_path, "numerosity_histogram"))

if __name__ == '__main__':

    start_time = time.time()

    #Command Line Arguments 
    parser = argparse.ArgumentParser(description='Run ANOVA on activations.csv')
    parser.add_argument('--model_directory', type=str, 
                        help='folder in data/models/ to find epoch folders in ')
    parser.add_argument('--layer', type=str,
                        help='Layer to save numerosity neurons for.')
    parser.add_argument('--dataset_directory', type=str,
                        help='Name of dataset directory to find numerosity neurons for.')
    parser.add_argument('--selection_method', type=str, choices=['variance','anova','anova1way'],
                        help='How to identify numerosity neurons. Options: variance, ie numerosity neurons are those for which numerosity explains more than 0.10 variance, other factors explain less than 0.01, as in (Stoianov and Zorzi); anova, ie numerosity neurons are those for which, in a two-way anova with numerosity and the other stimulus parameters as factors, the only significant association is with numerosity (Nieder); anova1way, ie numerosity neurons are those for which, in a two-way anova with numerosity and the other stimulus parameters as factors, numerosity is a significant association (regardless of the other parameters\' associations).')
    
    args = parser.parse_args()
    # reconcile arguments
    print('running with args:')
    print(args)

    # Get path to data directory
    models_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/models')

    dataset_path = os.path.join(models_path, args.model_directory, args.layer, args.dataset_directory)
    identifyNumerosityNeurons(dataset_path,args.selection_method)

    print('Total Run Time:')
    print("--- %s seconds ---" % (time.time() - start_time))
