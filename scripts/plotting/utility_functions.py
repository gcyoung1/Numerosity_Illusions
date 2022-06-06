import matplotlib.pyplot as plt

def saveNumerosityHistogram(method_path, max_activation_counts,numerosities):
    numerosity_neuron_counts = np.asarray([len(x) for x in sorted_numerosity_neurons])
    num_numerosity_neurons = sum(numerosity_neuron_counts)
    histFigure = plt.figure()
    histPlot = histFigure.add_subplot(1,1,1)
    percentages = 100*max_activation_counts/np.sum(max_activation_counts)  
    histPlot.bar(numerosities,percentages)
    histPlot.set_title(f'Numerosity Histogram')
    histPlot.set_ylabel('Percentage of Units')
    histPlot.set_xlabel('Preferred Numerosity')
    histFigure.savefig(method_path+'numerosity_histogram.jpg')
    plt.close(histFigure)

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

def createIndividualPlots(num_numerosities):
    # Return figure: a grid of subplots for each of num_numerosities numerosities
    subplot_dim = int(num_numerosities**(1/2))+1
    fig_side=subplot_dim*5
    fig, subplots = plt.subplots(subplot_dim,subplot_dim,figsize=(fig_side,fig_side))
    fig.suptitle(f"Average Tuning Curves",size='xx-large')
    subplots_list = np.ravel(subplots)
    return fig, subplots_list

def plotIndividualPlots(tuning_curves, std_errs,sorted_number_neurons,numerosities, subplots_list):
    # Plot the tuning curves on the list of subplots
    for i,idxs in enumerate(sorted_number_neurons):
        subplots_list[i].error_bar(numerosities,tuning_curves[i],yerr=std_errs[i], color='k')
        subplots_list[i].set_title(f"PN = {numerosities[i]} (n = {len(idxs)})")
    return subplots_list

def saveTuningCurves(tuning_curves, std_errs,sorted_number_neurons,numerosities):
    # Plot all tuning curves on same plot
    subplot_dim = int(num_numerosities**(1/2))+1
    fig_side=subplot_dim*5
    fig = plt.figure(figsize=(fig_side,fig_side))
    fig.suptitle(f"Average Tuning Curves",size='xx-large')

    for i,idxs in enumerate(sorted_number_neurons):
        fig.error_bar(numerosities,tuning_curves[i], yerr=std_errs[i]) 

    fig.legend(numerosities)
    return fig
