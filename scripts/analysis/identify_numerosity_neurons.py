import os
import time
import argparse
import pickle       
import numpy as np
import pandas as pd
import pingouin as pg
import random
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import re


from . import utility_functions as utils
from ..plotting import utility_functions as plotting_utils

def getAnovaDict(df,num_neurons,parameters_header):

    anova_dict = {}
    nonconverged_neurons = []
    for i in range(num_neurons):
        # Exclude from contention neurons with the same activation for all stimuli
        if len(df[f'n{i}'].unique()) > 1:
            #print(f"n{i}")
            start_time = time.time()
            anova_dict[f'n{i}'] = {}
            try:
                aov = pg.anova(dv=f'n{i}', between=parameters_header, data=df,detailed=True)

                anova_dict[f'n{i}']['converged'] = True
                # Add to dict 
                for row in range(len(aov)):
                    source = aov.at[row, 'Source']
                    anova_dict[f'n{i}'][source] = {}
                    anova_dict[f'n{i}'][source]['np2'] = aov.at[row,'np2']
                    anova_dict[f'n{i}'][source]['p-unc'] = aov.at[row,'p-unc']

            except np.linalg.LinAlgError as err:
                anova_dict[f'n{i}']['converged'] = False
                print(f"\n\n\nNeuron n{i} did not converge\n\n\n")
                nonconverged_neurons.append(f"n{i}")

            if i % 100 == 0:
                print(f"Anova took {time.time()-start_time} seconds, total will take {(num_neurons-i)*(time.time()-start_time)/60} more minutes")

    print(f"ANOVAs for the following neurons did not converge: {nonconverged_neurons}")
    return anova_dict

def getVarianceDict(df,num_neurons,parameters_header):

    variance_dict = {}
    nonconverged_neurons = []
    for i in range(num_neurons):
        # Exclude from contention neurons with the same activation for all stimuli
        if len(df[f'n{i}'].unique()) > 1:
            #print(f"n{i}")
            start_time = time.time()
            variance_dict[f'n{i}'] = {}
            #try:
            for source in parameters_header:
                x = np.log2(df[source].to_numpy()).reshape(-1,1)
                y = df[f'n{i}'].to_numpy().reshape(-1,1)
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5) 
                reg = LinearRegression().fit(x_train, y_train)
                variance_explained = reg.score(x_test, y_test)
                variance_dict[f'n{i}'][source] = {}
                variance_dict[f'n{i}'][source]['variance_explained'] = variance_explained

            variance_dict[f'n{i}']['converged'] = True

            # except np.linalg.LinAlgError as err:
            #     anova_dict[f'n{i}']['converged'] = False
            #     print(f"\n\n\nNeuron n{i} did not converge\n\n\n")
            #     nonconverged_neurons.append(f"n{i}")

            if i % 100 == 0:
                print(f"Regressions took {time.time()-start_time} seconds, total will take {(num_neurons-i)*(time.time()-start_time)/60} more minutes")

    print(f"Regressions for the following neurons did not converge: {nonconverged_neurons}")
    return variance_dict



def getNumerosityNeurons(info_dict,selection_method):
    non_numerosity_parameters = None
    numerosity_neurons = []

    significance_threshold = 0.01/len(list(info_dict.keys())) if 'corrected' in selection_method else 0.01

    for neuron_id in info_dict.keys():
        neuron_dict = info_dict[neuron_id]
        if not neuron_dict['converged']:
            continue
        if not non_numerosity_parameters:
            non_numerosity_parameters = list(neuron_dict.keys())
            non_numerosity_parameters = [x for x in non_numerosity_parameters if x not in ['converged','numerosity','Residual']]

        # Initialize assuming no effects
        numerosity_effects = False
        non_numerosity_effects = False

        if selection_method == 'anova' or selection_method == 'anova_corrected':

            # Effect is p-value
            numerosity_p_value = neuron_dict['numerosity']['p-unc']
            numerosity_effects = (numerosity_p_value < significance_threshold)
            # Non-numerosity parameters
            for parameter in non_numerosity_parameters:
                non_numerosity_p_value = neuron_dict[parameter]['p-unc']
                non_numerosity_effect = non_numerosity_p_value < 0.01
                non_numerosity_effects = non_numerosity_effects or non_numerosity_effect

        elif selection_method == 'anova1way' or selection_method == 'anova1way_corrected':
            # Only check for numerosity effect
            numerosity_p_value = neuron_dict['numerosity']['p-unc']
            numerosity_effects = (numerosity_p_value < significance_threshold)

        elif selection_method == 'variance':
            # Effect is partial eta-squared
            numerosity_variance = neuron_dict['numerosity']['np2']
            numerosity_effects = (numerosity_variance > 0.1)
            # Non-numerosity parameters
            for parameter in non_numerosity_parameters:
                non_numerosity_variance = neuron_dict[parameter]['np2']
                non_numerosity_effect = non_numerosity_variance > 0.01
                non_numerosity_effects = non_numerosity_effects or non_numerosity_effect

        elif selection_method == 'dewind_variance':
            # Effect is explained variance
            numerosity_variance = neuron_dict['numerosity']['variance_explained']
            numerosity_effects = (numerosity_variance > 0.1)
            # Non-numerosity parameters
            for parameter in non_numerosity_parameters:
                non_numerosity_variance = neuron_dict[parameter]['variance_explained']
                non_numerosity_effect = non_numerosity_variance > 0.01
                non_numerosity_effects = non_numerosity_effects or non_numerosity_effect

        if numerosity_effects and not non_numerosity_effects:
            numerosity_neurons.append(int(neuron_id[1:]))

    return np.array(numerosity_neurons)


def sortNumerosityNeurons(numerosity_neurons,numerosities,average_activations):
    max_activation_numerosity = average_activations.idxmax(axis=0)
    sorted_number_neurons = [[] for _ in range(len(numerosities))]
    for idx in numerosity_neurons:
        num = max_activation_numerosity[f'n{idx}']
        sorted_number_neurons[num].append(idx)
    return sorted_number_neurons

def identifyNumerosityNeurons(dataset_path,dataset_name,selection_method,percent_numerosity_metadata):
    method_path = os.path.join(dataset_path,f'{dataset_name}_{selection_method}_')

    df = utils.getActivationDataFrame(dataset_path,'activations')
    parameters_header = list(df.columns)

    num_neurons = len([x for x in list(df.columns) if re.match(r'n\d+', x)])
    parameters_header = [x for x in list(df.columns) if not re.match(r'n\d+', x)]

    num_nonzero_entries = df.astype(bool).sum(axis=0)
    if 'num_lines' in parameters_header and num_nonzero_entries['num_lines'] == 0:
        parameters_header.remove('num_lines')


    if not os.path.exists(os.path.join(dataset_path, 'numerosities.npy')):
        numerosities = df['numerosity'].unique().tolist()
        numerosities.sort()
        np.save(os.path.join(dataset_path, 'numerosities.npy'), numerosities)
    else:
        numerosities = np.load(os.path.join(dataset_path, 'numerosities.npy'))

    if selection_method == "dewind_variance":
        if not os.path.exists(os.path.join(dataset_path, 'variance_dict.pkl')):
            print("Performing regressions...")
            info_dict = getVarianceDict(df,num_neurons,parameters_header)
            with open(os.path.join(dataset_path, 'variance_dict.pkl'), 'wb') as f:
                pickle.dump(info_dict, f)
            f.close()
        else:
            print("Loading regressions...")
            with open(os.path.join(dataset_path, 'variance_dict.pkl'), 'rb') as f:
                info_dict = pickle.load(f)

    else:
        if not os.path.exists(os.path.join(dataset_path, 'anova_dict.pkl')):
            print("Performing anovas...")
            info_dict = getAnovaDict(df,num_neurons,parameters_header)
            with open(os.path.join(dataset_path, 'anova_dict.pkl'), 'wb') as f:
                pickle.dump(info_dict, f)
            f.close()
        else:
            print("Loading anovas...")
            with open(os.path.join(dataset_path, 'anova_dict.pkl'), 'rb') as f:
                info_dict = pickle.load(f)

    if not os.path.exists(method_path + 'numerosityneurons.npy'):
        print("Identifying numerosity neurons...")
        numerosity_neurons = getNumerosityNeurons(info_dict,selection_method)
        percent_of_layer = len(numerosity_neurons) / num_neurons
        line = percent_numerosity_metadata + "," + str(percent_of_layer) + "\n"

        percent_neurons_file = os.path.join(dataset_path, '..', '..', '..', 'percent_numerosity_neurons.csv')
        if not os.path.exists(percent_neurons_file):
            header = "model_name,layer,dataset_name,selection_method,percent_of_layer\n"
            with open(percent_neurons_file, 'a') as f:
                f.write(header)
        with open(percent_neurons_file, 'a') as f:
            f.write(line)

        print("Sorting numerosity neurons...")
        average_activations = df.groupby(['numerosity'],as_index=False).mean()
        sorted_numerosity_neurons = sortNumerosityNeurons(numerosity_neurons,numerosities,average_activations)
        # sorted_numerosity_neurons is an list of lists containing the indices of numerosity neurons for each
        # numerosity. Thus not all lists will be the same size so we can't save it as a regular numpy array
        # Hence dtype=object and allow_pickle=True below
        np.save(method_path + 'numerosityneurons', np.array(sorted_numerosity_neurons, dtype=object))
    else:
        average_activations = df.groupby(['numerosity'],as_index=False).mean()
        sorted_numerosity_neurons = np.load(method_path + 'numerosityneurons.npy', allow_pickle=True)
        numerosity_neurons = [idx for l in sorted_numerosity_neurons for idx in l]
    
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
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--model_directory', type=str, 
                        help='Directory in experiment directory to find dataset directories in ')
    parser.add_argument('--dataset_name', type=str,
                        help='Name of dataset to find numerosity neurons for.')
    parser.add_argument('--layer', type=str,
                        help='Layer to save numerosity neurons for.')
    parser.add_argument('--selection_method', type=str, choices=['variance', 'dewind_variance','anova','anova_corrected','anova1way','anova1way_corrected'],
                        help='How to identify numerosity neurons. Options: variance, ie numerosity neurons are those for which numerosity explains more than 0.10 variance using the partial eta squared of the anova, and other factors explain less than 0.01; dewind_variance, which has the same criteria but uses the variance explained by each factor in a single linear regression as in (Dewind 2019); anova, ie numerosity neurons are those for which, in a two-way anova with numerosity and the other stimulus parameters as factors, the only significant association is with numerosity (Nieder). In particular, we use a cutoff of 0.05 for non-numerosity significance, and a corrected p-value of 0.01 as the cutoff for numerosity significance; anova1way, ie numerosity neurons are those for which, in a two-way anova with numerosity and the other stimulus parameters as factors, numerosity is a significant association (regardless of the other parameters\' associations).')
    
    args = parser.parse_args()
    # reconcile arguments
    print('running with args:')
    print(args)

    # Get path to data directory
    models_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/models')

    dataset_path = os.path.join(models_path, args.experiment_name, args.model_directory, args.layer, args.dataset_name)
    percent_numerosity_metadata = f"{args.model_directory},{args.layer},{args.dataset_name},{args.selection_method}"
    identifyNumerosityNeurons(dataset_path,args.dataset_name,args.selection_method, percent_numerosity_metadata)

    print('Total Run Time:')
    print("--- %s seconds ---" % (time.time() - start_time))
