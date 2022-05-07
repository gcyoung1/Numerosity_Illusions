import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import sys
sys.path.append('../CORnet/')
import cornet
import os
sys.path.append('../models')
from thirtycnn import ThirtyCNN

def tensorToNumpy(tensor):
    return tensor.detach().cpu().numpy()

def set_bn_eval(module):
    if isinstance(module,nn.modules.batchnorm._BatchNorm):
        module.eval()

def getClassifierParams(model):
    params_to_train=[]
    for name,p in model.named_parameters():
        if "features" not in name:
            params_to_train.append(p)
    return params_to_train

def listToString(theList):
    return ','.join([str(x) for x in theList])

def createAccuracyCSV(folder,test_epochs):
    csvfile = open(os.path.join(folder,f'accuracy.csv'),'w+')
    header = listToString(test_epochs)
    csvfile.write('accuracy_type,'+header)
    csvfile.write('\n')
    csvfile.flush()
    return csvfile

def createActivationCSV(folder,dataset_name,features_size):
    csvfile = open(os.path.join(folder,f'{dataset_name}_activations.csv'),'w+')
    if "enumeration" in dataset_name or "barbell" in dataset_name:
        columnList = ["numerosity","condition"]
    elif "symbol" in dataset_name:
        columnList = ["number","font"]
    elif "solitaire" in dataset_name:
        columnList = ['rows','group','together','ratio']
    elif "dewind" in dataset_name:
        columnList = ['numerosity','square_side','bounding_side']
    header = listToString(columnList)
    csvfile.write(header+',')
    csvfile.write(listToString([f'n{i}' for i in range(features_size)]))
    csvfile.write('\n')
    csvfile.flush()
    return csvfile

def no_grad(model):
    for param in model.parameters():
        param.requires_grad = False

def initialize_model(model_name:Literal['alexnet','cornet_s'], pretrained:bool):
    if model_name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
        input_size = 224

    elif model_name == "cornet_s":
        model = cornet.cornet_s(pretrained=pretrained)
        input_size = 224

    else:
        raise ValueError("Model name not recognized")

    return model, input_size
    

