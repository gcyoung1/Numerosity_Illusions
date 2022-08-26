import matplotlib
from  matplotlib import pyplot as plt
matplotlib.use('Agg')
import numpy as np

def saveNumerosityHistogram(sorted_numerosity_neurons,numerosities):
    # Returns a figure showing the distribution of numerosity neurons
    # across the numerosities, by percentage of the total number of 
    # numerosity neurons
    numerosity_neuron_counts = np.asarray([len(x) for x in sorted_numerosity_neurons])
    percentages = 100*numerosity_neuron_counts/np.sum(numerosity_neuron_counts)  

    fig = plt.figure()
    fig.bar(numerosities,percentages)
    fig.title(f'Numerosity Neuron Percentage Histogram')
    fig.ylabel('Percentage of Units')
    fig.xlabel('Preferred Numerosity')
    return fig

def plotVarianceExplained(anova_dict, numerosity_neurons):
    # Create a subplot for each non-numerosity stimulus parameter where we'll plot
    # the variance explained by that parameter vs numerosity for each neuron
    # Numerosity neurons will be plotted in red, all others in black
    first_neuron = list(anova_dict.keys())[0]
    parameters_header = list(anova_dict[first_neuron].keys())
    fig, axs = plt.subplots(1,len(parameters_header)-1)
    axs[0].set_ylabel(f'Partial eta-squared {parameters_header[0]}')
    for row in range(1,len(parameters_header)):
        axs[row-1].set_xlabel(f'Partial eta-squared {parameters_header[row]}')
        axs[row-1].set_ylim(0,1)
        axs[row-1].set_xlim(0,1)
    
    for neuron_id in anova_dict.keys():
        # e.g. neuron_id = n0
        neuron_dict = anova_dict[neuron_id]
        numerosity_variance = neuron_dict['numerosity']['np2']
        # Plot it in red if it's a numerosity neuron, black otherwise
        if int(neuron_id[1:]) in numerosity_neurons:
            color = 'red'
        else:
            color = 'black'

        for row in range(1,len(parameters_header)):
            non_numerosity_variance = anova_dict[neuron_id][f'{parameters_header[row]}']['np2']
            axs[row-1].scatter(non_numerosity_variance,numerosity_variance,c=color)

    return fig

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
    # Takes in a subplot_list created by createIndividualPlots
    for i,idxs in enumerate(sorted_number_neurons):
        subplots_list[i].errorbar(numerosities,tuning_curves[i],yerr=std_errs[i], color='k')
        subplots_list[i].set_title(f"PN = {numerosities[i]} (n = {len(idxs)})")
    return subplots_list

def plotTuningOnePlot(tuning_curves, std_errs,sorted_number_neurons,numerosities):
    # Plot all tuning curves on same plot
    subplot_dim = int(len(numerosities)**(1/2))+1
    fig_side=subplot_dim*5
    fig = plt.figure(figsize=(fig_side,fig_side))
    fig.suptitle(f"Average Tuning Curves",size='xx-large')

    for i,idxs in enumerate(sorted_number_neurons):
        fig.error_bar(numerosities,tuning_curves[i], yerr=std_errs[i]) 

    fig.legend(numerosities)
    return fig
