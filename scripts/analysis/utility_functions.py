import numpy as np
import pandas as pd
import os

def getActivationDataFrame(path,filename):
    data = os.path.join(path,f'{filename}.csv')
    df = pd.read_csv(data)
    return df

def getTuningCurve(df, indices):
    import pdb;pdb.set_trace()
    indices = [f'n{x}' for x in indices]
    selectedColumns = df[indices]
    normalizedColumns = (selectedColumns-selectedColumns.min())/(selectedColumns.max()-selectedColumns.min())
    average = normalizedColumns.mean(axis=1)
    std_err = normalizedColumns.std(axis=1)/(normalizedColumns.shape[1])**(1/2)
    minActivation = average.min()
    maxActivation = average.max()
    activation_range = (maxActivation-minActivation)
    return (average-minActivation)/activation_range, std_err/activation_range

def getTuningCurves(sorted_number_neurons,numerosities,average_activations):
    tuning_curves = np.zeros((len(numerosities),len(numerosities)))
    std_errs = np.zeros((len(numerosities),len(numerosities)))

    for i,idxs in enumerate(sorted_number_neurons):
        tuningCurve, std_err = getTuningCurve(average_activations,idxs)    
        tuning_curves[i] = tuningCurve
        std_errs[i] = std_err
    return tuning_curves, std_errs
