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
    tuning_curves = np.zeros((len(numerosities),len(numerosities)))

    for i,idxs in enumerate(sorted_number_neurons):
        tuningCurve, std_err = utils.getAverageActivations(average_activations,idxs)    
        tuning_curves[i] = tuningCurve
        oneDPlots[i].error_bar(numerosities,tuningCurve,yerr=std_err, color='k')
        allSubplots.error_bar(numerosities,tuningCurve, yerr=std_err) 
        oneDPlots[i].set_title(f"PN = {numerosities[i]} (n = {len(idxs)})")

    np.save(method_path+"tuning_curves", tuning_curves)
    allSubplots.legend(numerosities)
    allFig.savefig(method_path+"All_TuningCurves.jpg")
    individual_fig.savefig(method_path+"Individual_TuningCurves.jpg")
    plt.close(allFig)
    plt.close(individual_fig)









    numerosity_neuron_counts = np.asarray([len(x) for x in sorted_numerosity_neurons])
    num_numerosity_neurons = sum(numerosity_neuron_counts)

    saveNumerosityHistogram(method_path, numerosity_neuron_counts,numerosities)

    print("Plotting tuning curves...")
    saveTuningCurves(method_path,sorted_numerosity_neurons,numerosities,average_activations,subplot_dim)


    subplot_dim = int(len(numerosities)**(1/2))+1






    if selection_method == 'variance':
        plotVarianceExplained(anova_dict, parameters_header, numerosity_neurons, layer_path)
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


