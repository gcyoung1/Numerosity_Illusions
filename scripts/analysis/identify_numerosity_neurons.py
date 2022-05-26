import os
import time
import argparse
import pickle       
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import random

import utility_functions as utils

def saveAnovaDict(df,num_neurons,parameters_header,save_path):
    anova_dict = {}
    nonzero_entries = df.astype(bool).sum(axis=0)

    # Don't do anova on num_lines if it's always 0
    if nonzero_entries['num_lines'] == 0:
        parameters_header.remove('num_lines')

    for i in range(num_neurons):
        # Exclude from contention neurons with 0 activation for all stimuli
        if nonzero_entries[f"n{i}"]:
            print(f"n{i}")
            import pdb;pdb.set_trace()
            aov = pg.anova(dv=f'n{i}', between=parameters_header, data=df,detailed=True)
            
            # Add to dict 
            anova_dict[f'n{i}'] = {}
            for row in range(len(parameters_header)):
                anova_dict[f'n{i}'][f'{parameters_header[row]}'] = {}
                anova_dict[f'n{i}'][f'{parameters_header[row]}']['np2'] = aov.at[row,'np2']
                anova_dict[f'n{i}'][f'{parameters_header[row]}']['p-unc'] = aov.at[row,'p-unc']

    f = open(os.path.join(save_path, 'anova_dict'), 'wb')
    pickle.dump(anova_dict, f)
    f.close()
    return anova_dict


def getNumerosityNeurons(anova_dict,parameters_header,selection_method):
    numerosity_neurons = []

    for neuron_id in anova_dict.keys():
        neuron_dict = anova_dict[f'n{i}']

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

def plotVarianceExplained(anova_dict, parameters_header, numerosity_neurons, layer_path):
    # Create a subplot for each non-numerosity stimulus parameter where we'll plot
    # the variance explained by that parameter vs numerosity
    fig, axs = plt.subplots(1,len(parameters_header)-1)
    axs[0].set_ylabel(f'Partial eta-squared {parameters_header[0]}')
    for row in range(1,len(parameters_header)):
        axs[row-1].set_xlabel(f'Partial eta-squared {parameters_header[row]}')
        axs[row-1].set_ylim(0,1)
        axs[row-1].set_xlim(0,1)
    
    for neuron_id in anova_dict.keys():
        neuron_dict = anova_dict[f'n{i}']
        numerosity_variance = neuron_dict['numerosity']['np2']
        # Plot it in red if it's a numerosity neuron, black otherwise
        if int(neuron_id[1:]) in numerosity_neurons:
            color = 'red'
        else:
            color = 'black'

        for row in range(1,len(parameters_header)):
            non_numerosity_variance = anova_dict[f'n{i}'][f'{parameters_header[row]}']
            axs[row-1].scatter(non_numerosity_variance,numerosity_variance,c=color)

    fig.savefig(os.path.join(layer_path,"variance_explained.jpg"))
    plt.close(fig)


def sortNumerosityNeurons(numerosity_neurons,numerosities,average_activations):
    max_activation_numerosity = average_activations.idxmax(axis=0)
    sorted_number_neurons = [[] for _ in range(len(numerosities))]
    for idx in numerosity_neurons:
        num = max_activation_numerosity[f'n{idx}']
        sorted_number_neurons[num].append(idx)
    return sorted_number_neurons

def saveNumerosityHistogram(method_path, max_activation_counts,numerosities):
    histFigure = plt.figure()
    histPlot = histFigure.add_subplot(1,1,1)
    percentages = 100*max_activation_counts/np.sum(max_activation_counts)  
    histPlot.bar(numerosities,percentages)
    histPlot.set_title(f'Numerosity Histogram')
    histPlot.set_ylabel('Percentage of Units')
    histPlot.set_xlabel('Preferred Numerosity')
    histFigure.savefig(method_path+'numerosity_histogram.jpg')
    plt.close(histFigure)

def saveTuningCurves(method_path,sorted_number_neurons,numerosities,average_activations,subplot_dim):
    fig_side=subplot_dim*5
    individual_fig, individual_subplots = plt.subplots(subplot_dim,subplot_dim,figsize=(fig_side,fig_side))
    allFig, allSubplots = plt.subplots(figsize=(fig_side,fig_side))
    individual_fig.suptitle(f"Average Tuning Curves",size='xx-large')
    allFig.suptitle(f"Average Tuning Curves",size='xx-large')
    oneDPlots = np.ravel(individual_subplots)
    tuning_curve_matrix = np.zeros((len(numerosities),len(numerosities)))

    for i,idxs in enumerate(sorted_number_neurons):
        tuningCurve, std_err = utils.getAverageActivations(average_activations,idxs)    
        tuning_curve_matrix[i] = tuningCurve
        oneDPlots[i].error_bar(numerosities,tuningCurve,yerr=std_err, color='k')
        allSubplots.error_bar(numerosities,tuningCurve, yerr=std_err) 
        oneDPlots[i].set_title(f"PN = {numerosities[i]} (n = {len(idxs)})")

    np.save(method_path+"tuning_matrix", tuning_curve_matrix)
    allSubplots.legend(numerosities)
    allFig.savefig(method_path+"All_TuningCurves.jpg")
    individual_fig.savefig(method_path+"Individual_TuningCurves.jpg")
    plt.close(allFig)
    plt.close(individual_fig)

def identifyNumerosityNeurons(layer_path,selection_method):
    method_path = os.path.join(layer_path,f'{selection_method}_')
    parameters_header = ['numerosity', 'size','spacing','num_lines']

    df = utils.getActivationDataFrame(layer_path,f'activations')

    num_neurons = len(df.columns)-len(parameters_header)
    numerosities = df['numerosity'].unique().tolist()
    numerosities.sort()

    if not os.path.exists(os.path.join(layer_path, 'anova_dict.pkl')):
        print("Performing anovas...")
        anova_dict = saveAnovaDict(df,num_neurons,parameters_header,layer_path)
    else:
        print("Loading anovas...")
        anova_dict = pickle.load(os.path.join(save_path, 'anova_dict.pkl'))

    print("Identifying numerosity neurons...")
    numerosity_neurons = getNumerosityNeurons(anova_dict,parameters_header,selection_method)

    print("Sorting numerosity neurons...")
    average_activations = df.groupby(['numerosity'],as_index=False).mean()
    sorted_numerosity_neurons = sortNumerosityNeurons(numerosity_neurons,numerosities,average_activations)
    np.save(method_path + 'numerosityneurons', np.array(sorted_numerosity_neurons))
    
    numerosity_neuron_counts = np.asarray([len(x) for x in sorted_numerosity_neurons])
    num_numerosity_neurons = sum(numerosity_neuron_counts)

    subplot_dim = int(len(numerosities)**(1/2))+1

    print("Plotting tuning curves...")
    saveTuningCurves(method_path,sorted_numerosity_neurons,numerosities,average_activations,subplot_dim)
    saveNumerosityHistogram(method_path, numerosity_neuron_counts,numerosities)
    if selection_method == 'variance':
        plotVarianceExplained(anova_dict, parameters_header, numerosity_neurons, layer_path)

if __name__ == '__main__':

    start_time = time.time()

        #Command Line Arguments 
    parser = argparse.ArgumentParser(description='Run ANOVA on activations.csv')
    parser.add_argument('--model_directory', type=str, 
                        help='folder in data/models/ to find epoch folders in ')
    parser.add_argument('--dataset_directory', type=str,
                        help='Name of dataset directory in data/models/args.model_directory to find numerosity neurons of.')
    parser.add_argument('--layers', nargs='+',
                        help='Names of directories in data/models/args.model_directory/args.dataset_directory containing csv files')
    parser.add_argument('--selection_method', type=str,
                        help='How to identify numerosity neurons. Options: variance, ie numerosity neurons are those for which numerosity explains more than 0.10 variance, other factors explain less than0.01, as in (Stoianov and Zorzi); anova, ie numerosity neurons are those for which, in a two-way anova with numerosity and the other stimulus parameters as factors, the only significant association is with numerosity (Nieder); anova1way, ie numerosity neurons are those for which, in a two-way anova with numerosity and the other stimulus parameters as factors, numerosity is a significant association (regardless of the other parameters\' associations).')
    
    args = parser.parse_args()
    # reconcile arguments
    print('running with args:')
    print(args)


    
    # Get path to data directory
    models_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/models')

    dataset_path=os.path.join(models_path, args.model_directory, args.dataset_directory)

    for layer in args.layers:
        print(f"Layer {layer}")
        layer_path = os.path.join(dataset_path,layer)
        identifyNumerosityNeurons(layer_path,args.selection_method)

    print('Total Run Time:')
    print("--- %s seconds ---" % (time.time() - start_time))
