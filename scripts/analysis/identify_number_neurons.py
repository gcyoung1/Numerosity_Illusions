import os
import time
import argparse
import pickle       
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn
import random

def getActivationDataFrame(path,filename):
    data = os.path.join(path,f'{filename}.csv')
    df = pd.read_csv(data)
    return df

def getNumerosityNeurons(df,num_neurons,parameters_header,selection_method):
    sums = df.sum(axis=0)
    numerosity_neurons = []

    for i in range(num_neurons):
        # Exclude from contention neurons with 0 activation for all stimuli
        if sums[f'n{i}'] != 0:
            aov = pg.anova(dv=f'n{i}', between=parameters_header, data=df,detailed=True)
            numerosity_effects = False
            non_numerosity_effects = False

            if selection_method == 'anova':
                numerosity_effects = (aov.at[0,'p-unc'] < 0.01)
                for row in range(1,len(parameters_header)):
                    non_numerosity_effect = aov.at[row,'p-unc'] < 0.01
                    non_numerosity_effects = non_numerosity_effects or non_numerosity_effect

            elif selection_method == 'anova1way':
                numerosity_effects = (aov.at[0,'p-unc'] < 0.01)

            elif selection_method == 'variance':
                variance_dict[f'n{i}'] = {}
                numerosity_variance = aov.at[0,'np2']
                numerosity_effects = (numerosity_variance > 0.1)
                variance_dict[f'n{i}']['numerosity'] = numerosity_variance
                for row in range(1,len(parameters_header)):
                    non_numerosity_effect = aov.at[row,'np2'] > 0.1
                    variance_dict[f'n{i}'][f'{parameters_header[row]}'] = aov.at[row,'np2']
                    non_numerosity_effects = non_numerosity_effects or non_numerosity_effect

            if numerosity_effects and not non_numerosity_effects:
                numerosity_neurons.append(i)
    return numerosity_neurons

def sortNumerosityNeurons(numerosity_neurons,numerosities,average_activations_all_conditions):
    max_activation_numerosity = average_activations_all_conditions.idxmax(axis=0)
    sortedNumberNeurons = [[] for _ in range(len(numerosities))]
    for idx in numerosity_neurons:
        num = max_activation_numerosity[f'n{idx}']
        sortedNumberNeurons[num].append(idx)
    return sortedNumberNeurons

def saveRandomTuningCurvesPlots(savePath,average_activations_each_condition,average_activations_all_conditions,conditions,conditionHeader,numerosities, sortedNumberNeurons,subplotDim):
    figSide=subplotDim*5
    fig, splots = plt.subplots(subplotDim,subplotDim,figsize=(figSide,figSide))
    oneDPlots=np.ravel(splots)
    
    fig.suptitle(f"Random Tuning Curves")

    for i,indices in enumerate(sortedNumberNeurons):
        if indices:
            idx = random.choice(indices)
            for cond in conditions:
                tuningCurve = average_activations_each_condition[average_activations_each_condition[conditionHeader] == cond][f'n{idx}']
                oneDPlots[i].plot(numerosities,tuningCurve) 

            avTuningCurve = average_activations_all_conditions[f'n{idx}']
            oneDPlots[i].plot(numerosities, avTuningCurve, 'k')
            oneDPlots[i].set_xlabel('Numerosity')
            oneDPlots[i].set_title(f"PN = {numerosities[i]} Neuron {idx}")
           
    handles, labels = plt.gca().get_legend_handles_labels()        
    conditions.append('Average Over All Conditions')
    fig.legend(handles,conditions,loc='lower right')         
    fig.tight_layout()
    fig.savefig(savePath+"Random_TuningCurves")
    plt.close(fig)


def saveNumerosityHistogram(savePath, max_activation_counts,numerosities):
    histFigure = plt.figure()
    histPlot = histFigure.add_subplot(1,1,1)
    percentages = 100*max_activation_counts/np.sum(max_activation_counts)  
    histPlot.bar(numerosities,percentages)
    histPlot.set_title(f'Numerosity Histogram')
    histPlot.set_ylabel('Percentage of Units')
    histPlot.set_xlabel('Preferred Numerosity')
    histFigure.savefig(savePath+'numerosity_histogram.jpg')
    plt.close(histFigure)

def getAverageActivations(df, indices):
    indices = [f'n{x}' for x in indices]
    selectedColumns = df[indices]
    average = selectedColumns.mean(axis=1)
    minActivation = average.min()
    maxActivation = average.max()
    return (average-minActivation)/(maxActivation-minActivation)


def saveAverageTuningCurves(savePath,sortedNumberNeurons,numerosities,average_activations_all_conditions,subplotDim):
    figSide=subplotDim*5
    fig, splots = plt.subplots(subplotDim,subplotDim,figsize=(figSide,figSide))
    allFig, allPlot = plt.subplots(figsize=(figSide,figSide))
    fig.suptitle(f"Average Tuning Curves",size='xx-large')
    allFig.suptitle(f"Average Tuning Curves",size='xx-large')
    oneDPlots = np.ravel(splots)
    tuning_curve_matrix = np.zeros((len(numerosities),len(numerosities)))

    for i,idxs in enumerate(sortedNumberNeurons):
        tuningCurve = getAverageActivations(average_activations_all_conditions,idxs)        
        tuning_curve_matrix[i] = tuningCurve
        oneDPlots[i].plot(numerosities,tuningCurve,color='k')
        allPlot.plot(numerosities,tuningCurve) 
        oneDPlots[i].set_title(f"PN = {numerosities[i]} (n = {len(idxs)})")

    np.save(savePath+"tuning_matrix", tuning_curve_matrix)
    allPlot.legend(numerosities)
    allFig.savefig(savePath+"All_TuningCurves.jpg")
    fig.savefig(savePath+"Individual_TuningCurves.jpg")
    plt.close(allFig)
    plt.close(fig)

def anova(dataset_name):
    savePath = os.path.join(args.layer_folder,f'{dataset_name}_{args.selection_criteria}_')
    parameters_header = ['numerosity', 'size','spacing','num_lines']

    df = getActivationDataFrame(args.layer_folder,f'{dataset_name}_activations')

    num_neurons = len(df.columns)-len(parameters_header)
    numerosities = df['numerosity'].unique().tolist()
    numerosities.sort()

    numerosity_neurons = getNumerosityNeurons(df,num_neurons,parameters_header)

    if selection_method == 'variance':
        variance_dict = {}
        fig, axs = plt.subplots(1,len(parameters_header)-1)
        axs[0].set_ylabel(f'Partial eta-squared {parameters_header[0]}')
        for row in range(1,len(parameters_header)):
            axs[row-1].set_xlabel(f'Partial eta-squared {parameters_header[row]}')
            axs[row-1].set_ylim(0,1)
            axs[row-1].set_xlim(0,1)

    if args.selection_criteria == 'variance':
        fig.savefig(savePath+"variance_plots")
        f = open(savePath+'variance_dict', 'wb')
        pickle.dump(variance_dict, f)
        f.close()



                if numerosity_effects and not non_numerosity_effects:
                    for row in range(1,len(parameters_header)):
                        axs[row-1].scatter(variance_dict[f'n{i}'][f'{parameters_header[row]}'],numerosity_variance,c='red')
                        axs[row-1].set_aspect('equal', adjustable='box')
                else:
                    for row in range(1,len(parameters_header)):
                        axs[row-1].scatter(variance_dict[f'n{i}'][f'{parameters_header[row]}'],numerosity_variance,c='black')






    sortedNumberNeurons = sortNumerosityNeurons(numerosity_neurons,numerosities,average_activations_all_conditions)
    np.save(savePath+'numberneurons', np.array(sortedNumberNeurons))
    
    max_activation_counts = np.asarray([len(x) for x in sortedNumberNeurons])

    subplotDim = int(len(numerosities)**(1/2))+1

    if not "dewind" in dataset_name:
        saveRandomTuningCurvesPlots(savePath,average_activations_each_condition,average_activations_all_conditions,conditions,parameters_header[1],numerosities,sortedNumberNeurons,subplotDim)

    saveAverageTuningCurves(savePath,sortedNumberNeurons,numerosities,average_activations_all_conditions,subplotDim)

    saveNumerosityHistogram(savePath, max_activation_counts,numerosities)

    num_number_neurons = sum([len(x) for x in sortedNumberNeurons])
    return 100*num_number_neurons/num_neurons



if __name__ == '__main__':

    start_time = time.time()

        #Command Line Arguments 
    parser = argparse.ArgumentParser(description='Run ANOVA on activations.csv')
    parser.add_argument('--input-data', type=str, default='', metavar='I',
                        help='folder in /outputs/ to find epoch folders in ')
    parser.add_argument('--selection-criteria', type=str, default='anova', metavar='I',
                        help='How to identify number neurons. Options: variance, ie variance numerosity explains more than 0.10 variance, other factors explain less than 0.10, as in (Stoianov and Zorzi); anova, ie a two-way anova with numerosity and stimulus type as factors where the only significant association is with numerosity (Nieder); anova1way, ie a 1-way anova with just numerosity as a factor.')
    parser.add_argument('--dataset-names', nargs='+', default=[], metavar='I',
                        help='Names of dataset folders in /stimuli')
    parser.add_argument('--layers', nargs='+', default=[], metavar='I',
                        help='Names of directories in epoch folders containing csv files')
    


    args = parser.parse_args()
    # reconcile arguments
    print('running with args:')
    print(args)
    
    path=os.path.join('../data/outputs/', args.input_data)
    csvtypeDict={}
    for layer in args.layers:
        csvtypeDict[layer]={}
        for dataset_name in args.dataset_names:
            csvtypeDict[layer][dataset_name] = []

    epochs = []
    for subfolder in sorted(os.listdir(path)):
      epochFolder = os.path.join(path,subfolder)
      if os.path.isdir(epochFolder) and 'epoch' in epochFolder:
        epoch = epochFolder.split('epoch')[-1]
        print(f"Epoch {epoch}")
        epochs.append(int(epoch))
        for layer in args.layers:
            args.layer_folder = os.path.join(epochFolder,layer)
            for dataset_name in args.dataset_names:
                csvtypeDict[layer][dataset_name].append(anova(dataset_name))

    
    for layer in args.layers:
        for dataset_name in args.dataset_names:
            np.save(os.path.join(path,f"{layer}_{dataset_name}_percent_numerosity_neurons"), np.array(csvtypeDict[layer][dataset_name]))
            print(f"Layer {layer} percent {dataset_name} neurons history: {csvtypeDict[layer][dataset_name]}")
            plt.plot(epochs, csvtypeDict[layer][dataset_name])
            plt.ylim(0,100)
            plt.ylabel("Percent Numerosity Neurons")
            plt.xlabel("Epoch")
            plt.savefig(os.path.join(path,f"{layer}_{dataset_name}_percent_numerosity_neurons.jpg"))
            plt.close()

    print('Total Run Time:')
    print("--- %s seconds ---" % (time.time() - start_time))
