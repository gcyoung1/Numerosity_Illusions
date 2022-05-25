import os
from torchvision import models

def tensorToNumpy(tensor):
    return tensor.detach().cpu().numpy()

def getClassifierParams(model):
    params_to_train=[]
    for name,p in model.named_parameters():
        if "features" not in name:
            params_to_train.append(p)
    return params_to_train

def listToString(l:list):
    return ','.join([str(x) for x in l])

def writeAndFlush(csv_file, line:str):
    csv_file.write(line + '\n')
    csv_file.flush()

def createActivationCSV(folder,features_size:int):
    column_list = ['numerosity','size','spacing','num_lines']
    parameters_header = listToString(column_list)
    neuron_names = [f'n{i}' for i in range(features_size)]
    activations_header = listToString(neuron_names)
    header = parameters_header + ',' + activations_header

    csv_file = open(os.path.join(folder,f'activations.csv'),'w+')
    writeAndFlush(csv_file, header)
    return csv_file

def noGrad(model):
    for param in model.parameters():
        param.requires_grad = False

def initializeModel(model_name:str, pretrained:bool):
    if model_name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
        input_size = 224

    elif model_name == "cornet_s":
        raise NotImplementedError("I haven't added the CORnet code yet.")
        model = cornet.cornet_s(pretrained=pretrained)
        input_size = 224

    else:
        raise ValueError("Model name not recognized")

    noGrad(model)
    return model, input_size
    

