import matplotlib
from  matplotlib import pyplot as plt
matplotlib.use('Agg')
import numpy as np

def saveNumerosityHistogram(sorted_numerosity_neurons,numerosities):
    # Returns a figure showing the distribution of numerosity neurons
    # across the numerosities, by percentage of the total number of 
    # numerosity neurons
    numerosity_neuron_counts = np.asarray([len(x) for x in sorted_numerosity_neurons])
    print(f"numerosity_neuron_counts: {numerosity_neuron_counts}")
    percentages = 100*numerosity_neuron_counts/np.sum(numerosity_neuron_counts)  

    fig = plt.figure()
    plt.bar(numerosities,percentages)
    plt.title(f'Numerosity Neuron Percentage Histogram')
    plt.ylabel('Percentage of Units')
    plt.xlabel('Preferred Numerosity')
    return fig

def plotVarianceExplained(anova_dict, numerosity_neurons):
    # Create a subplot for each non-numerosity stimulus parameter where we'll plot
    # the variance explained by that parameter vs numerosity for each neuron
    # Numerosity neurons will be plotted in red, all others in black
    first_neuron = list(anova_dict.keys())[0]
    non_numerosity_parameters = list(anova_dict[first_neuron].keys())
    non_numerosity_parameters.remove('converged')
    non_numerosity_parameters.remove('numerosity')

    fig, axs = plt.subplots(1,len(non_numerosity_parameters))
    axs[0].set_ylabel(f'Partial eta-squared numerosity')
    for i in range(len(non_numerosity_parameters)):
        axs[i].set_xlabel(f'Partial eta-squared {non_numerosity_parameters[i]}')
        axs[i].set_ylim(0,1)
        axs[i].set_xlim(0,1)
    
    numerosity_variances = []
    colors = []
    non_numerosity_variances = [[] for i in range(len(non_numerosity_parameters))]
    for neuron_id in anova_dict.keys():
        # e.g. neuron_id = n0
        neuron_dict = anova_dict[neuron_id]
        if not neuron_dict['converged']:
            continue
        numerosity_variances.append(neuron_dict['numerosity']['np2'])
        # Plot it in red if it's a numerosity neuron, black otherwise
        if int(neuron_id[1:]) in numerosity_neurons:
            colors.append('red')
        else:
            colors.append('black')

        for i in range(len(non_numerosity_parameters)):
            non_numerosity_variances[i].append(anova_dict[neuron_id][f'{non_numerosity_parameters[i]}']['np2'])

    for i in range(len(non_numerosity_parameters)):
        dot_sizes = [0.5]*len(numerosity_variances)
        axs[i].scatter(non_numerosity_variances[i],numerosity_variances,c=colors, s=dot_sizes)

    return fig

def createIndividualPlots(num_numerosities):
    # Return figure: a grid of subplots for each of num_numerosities numerosities
    subplot_dim = int(num_numerosities**(1/2))+1
    fig_side=subplot_dim*5
    fig, subplots = plt.subplots(subplot_dim,subplot_dim,figsize=(fig_side,fig_side))
    fig.suptitle(f"Average Tuning Curves",size='xx-large')
    subplots_list = np.ravel(subplots)
    return fig, subplots_list

def plotIndividualPlots(tuning_curves, std_errs,sorted_number_neurons,numerosities, subplots_list, color, label):
    # Plot the tuning curves on the list of subplots
    # Takes in a subplot_list created by createIndividualPlots
    for i,idxs in enumerate(sorted_number_neurons):
        subplots_list[i].errorbar(numerosities,tuning_curves[i],yerr=std_errs[i], color=color, label=label)
        subplots_list[i].set_title(f"PN = {numerosities[i]} (n = {len(idxs)})")
    subplots_list[0].legend()
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
