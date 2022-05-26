import pandas as pd

def getActivationDataFrame(path,filename):
    data = os.path.join(path,f'{filename}.csv')
    df = pd.read_csv(data)
    return df

def getAverageActivations(df, indices):
    indices = [f'n{x}' for x in indices]
    selectedColumns = df[indices]
    average = selectedColumns.mean(axis=1)
    std_err = selectedColumns.std(axis=1)/(selectedColumns.shape[0])**(1/2)
    minActivation = average.min()
    maxActivation = average.max()
    activation_range = (maxActivation-minActivation)
    return (average-minActivation)/activation_range, std_err/activation_range
