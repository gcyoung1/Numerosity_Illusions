import matplotlib.pyplot as plt
import time
import argparse
import os
import numpy as np

from . import utility_functions as utils

if __name__ == '__main__':

    start_time = time.time()

    #Command Line Arguments 
    parser = argparse.ArgumentParser(description='Plot the tuning curves on a particular dataset.')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Name of experiment')
    parser.add_argument('--model_directory', type=str, required=True,
                        help='Directory in experiment directory to find dataset directories in')
    parser.add_argument('--layer', type=str, required=True,
                        help='Layer to save tuning curves for.')
    parser.add_argument('--numerosity_neurons_dataset_name', type=str, required=True,
                        help='Name of dataset directory to look for numerosity neurons in. This determines which neurons in the layer have tuning curves saved for them.')
    parser.add_argument('--selection_method', type=str, choices=['variance','anova','anova1way'], required=True,
                        help='Within the numerosity_neurons_dataset_name, which selection method to use the numerosity neurons of.')
    parser.add_argument('--activations_dataset_name', type=str, required=True,
                        help='Name of dataset directory to use the activations of. This determines which activations (ie which dataset the activations are in response to) are used to create the tuning curves for the numerosity neurons specified above.')
    
    args = parser.parse_args()
    # reconcile arguments
    print('running with args:')
    print(args)

    # Get path to model directory
    models_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/models', args.experiment_name, args.model_directory, args.layer)

    # Load numerosity neurons
    numerosity_neuron_path = os.path.join(models_path, args.numerosity_neurons_dataset_name)
    numerosities = np.load(os.path.join(numerosity_neuron_path, 'numerosities.npy'))
    # Allow pickle since subarrays are different lengths
    sorted_numerosity_neurons = np.load(os.path.join(numerosity_neuron_path,f"{args.selection_method}_numerosityneurons.npy"), allow_pickle=True)

    # Load tuning curves
    numerosity_neuron_path = os.path.join(models_path, args.numerosity_neurons_dataset_name)
    nonillusory_tuning_curves = np.load(os.path.join(numerosity_neuron_path, f"{args.numerosity_neurons_dataset_name}_{args.selection_method}_tuning_curves.npy"))
    nonillusory_std_errs = np.load(os.path.join(numerosity_neuron_path, f"{args.numerosity_neurons_dataset_name}_{args.selection_method}_std_errs.npy"))

    activations_path = os.path.join(models_path, args.activations_dataset_name)
    illusory_tuning_curves = np.load(os.path.join(activations_path, f"{args.numerosity_neurons_dataset_name}_{args.selection_method}_tuning_curves.npy"))
    illusory_std_errs = np.load(os.path.join(activations_path, f"{args.numerosity_neurons_dataset_name}_{args.selection_method}_std_errs.npy"))

    # Save the tuning curves of the numerosity neurons on these activations
    fig, subplots_list = utils.createIndividualPlots(len(numerosities))
    # Plot nonillusory tuning curves
    subplots_list = utils.plotIndividualPlots(nonillusory_tuning_curves, nonillusory_std_errs,sorted_numerosity_neurons,numerosities, subplots_list, label=args.numerosity_neurons_dataset_name, color='k')
    # Plot illusory tuning curves
    _ = utils.plotIndividualPlots(illusory_tuning_curves, illusory_std_errs,sorted_numerosity_neurons,numerosities, subplots_list, label=args.activations_dataset_name,color='red')

    save_path = os.path.join(models_path, f"{args.activations_dataset_name}_{args.selection_method}_")
    fig.savefig(save_path+'plots')

    print('Total Run Time:')
    print("--- %s seconds ---" % (time.time() - start_time))
