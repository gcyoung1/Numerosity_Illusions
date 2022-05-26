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

def saveTuningCurves(method_path,sorted_number_neurons,numerosities,average_activations,subplot_dim):
    fig_side=subplot_dim*5
    individual_fig, individual_subplots = plt.subplots(subplot_dim,subplot_dim,figsize=(fig_side,fig_side))
    allFig, allSubplots = plt.subplots(figsize=(fig_side,fig_side))
    individual_fig.suptitle(f"Average Tuning Curves",size='xx-large')
    allFig.suptitle(f"Average Tuning Curves",size='xx-large')
    oneDPlots = np.ravel(individual_subplots)
    tuning_curve_matrix = np.zeros((len(numerosities),len(numerosities)))

    for i,idxs in enumerate(sorted_number_neurons):
        tuningCurve, std_err = getAverageActivations(average_activations,idxs)    
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

    df = getActivationDataFrame(layer_path,f'activations')

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
