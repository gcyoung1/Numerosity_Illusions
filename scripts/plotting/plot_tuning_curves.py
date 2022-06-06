import matplotlib.pyplot as plt

def saveTuningCurves(tuning_curves, std_errs,sorted_number_neurons,numerosities):
    allFig, allSubplots = plt.subplots(figsize=(fig_side,fig_side))
    allFig.suptitle(f"Average Tuning Curves",size='xx-large')

    for i,idxs in enumerate(sorted_number_neurons):
        allSubplots.error_bar(numerosities,tuning_curves[i], yerr=std_errs[i]) 

    allSubplots.legend(numerosities)
    allFig.savefig(method_path+"All_TuningCurves.jpg")
    plt.close(allFig)



if __name__ == '__main__':

    start_time = time.time()

    #Command Line Arguments 
    parser = argparse.ArgumentParser(description='Plot the tuning curves on a particular dataset.')
    parser.add_argument('--model_directory', type=str, required=True,
                        help='folder in data/models/ to find epoch folders in ')
    parser.add_argument('--layer', type=str, required=True,
                        help='Layer to save tuning curves for.')
    parser.add_argument('--numerosity_neurons_dataset_directory', type=str, required=True,
                        help='Name of dataset directory to look for numerosity neurons in. This determines which neurons in the layer have tuning curves saved for them.')
    parser.add_argument('--selection_method', type=str, choices=['variance','anova','anova1way'], required=True,
                        help='Within the numerosity_neurons_dataset_directory, which selection method to use the numerosity neurons of.')
    parser.add_argument('--activations_dataset_directory', type=str, required=True,
                        help='Name of dataset directory to use the activations of. This determines which activations (ie which dataset the activations are in response to) are used to create the tuning curves for the numerosity neurons specified above.')
    
    args = parser.parse_args()
    # reconcile arguments
    print('running with args:')
    print(args)

    # Get path to model directory
    models_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/models', args.model_directory, args.layer)

    # Load numerosity neurons
    numerosity_neuron_path = os.path.join(models_path, args.numerosity_neurons_dataset_directory)
    numerosities = np.load(os.path.join(numerosity_neuron_path, 'numerosities.npy'))
    sorted_numerosity_neurons = np.load(os.path.join(numerosity_neuron_path,f"{args.selection_method}_numerosityneurons.npy"))

    # Load activations
    activations_path = os.path.join(models_path, args.activations_dataset_directory)
    df = utils.getActivationDataFrame(activations_path,'activations')
    average_activations = df.groupby(['numerosity'],as_index=False).mean()

    # Save the tuning curves of the numerosity neurons on these activations
    save_path = os.path.join(activations_path, f"{args.numerosity_neurons_dataset_directory}_{args.selection_method}_")
    tuning_curves, std_errs = utils.getTuningCurves(sorted_numerosity_neurons,numerosities,average_activations)
    np.save(save_path+"tuning_curves", tuning_curves)
    np.save(save_path+"std_errs", std_errs)

    print('Total Run Time:')
    print("--- %s seconds ---" % (time.time() - start_time))
