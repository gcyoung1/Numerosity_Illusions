import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import sys
sys.path.append('../CORnet/')
import cornet
import os
sys.path.append('../models')
from identity import Identity
from hook import Hook
from finetune import Finetune
from thirtycnn import ThirtyCNN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def tensorToNumpy(tensor):
    return tensor.detach().cpu().numpy()

def set_bn_eval(module):
    if isinstance(module,nn.modules.batchnorm._BatchNorm):
        module.eval()

def makeConfusionPlots(folder,responseConfusionMatrix,correctSoftmaxConfusionMatrix,incorrectSoftmaxConfusionMatrix,totalSoftmaxConfusionMatrix):
    num_classes = len(responseConfusionMatrix)
    subplotDim = int((num_classes)**(1/2))+1
    figDim = subplotDim*5
    responseFig, responsePlots = plt.subplots(subplotDim,subplotDim,figsize=(figDim,figDim))
    flatResponsePlots=np.ravel(responsePlots)
    responseFig.suptitle(f"Response Confusion Graphs")

    correctSoftmaxFig, correctSoftmaxPlots = plt.subplots(subplotDim,subplotDim,figsize=(figDim,figDim))
    flatCorrectSoftmaxPlots=np.ravel(correctSoftmaxPlots)
    correctSoftmaxFig.suptitle(f"Correct v Incorrect Softmax Confusion Graphs")

    softmaxFig, softmaxPlots = plt.subplots(subplotDim,subplotDim,figsize=(figDim,figDim))
    flatSoftmaxPlots=np.ravel(softmaxPlots)
    softmaxFig.suptitle(f"Softmax Confusion Graphs")

    for i in range(num_classes):
        flatResponsePlots[i].plot(range(1,num_classes+1),responseConfusionMatrix[i])
        flatResponsePlots[i].set_ylim(0,100)
        flatResponsePlots[i].set_ylabel("Percent of Responses")
        flatResponsePlots[i].set_xlabel("Numerosity")
        flatResponsePlots[i].set_title(f"Reponses to Numerosity {i+1}")
  
        flatSoftmaxPlots[i].plot(range(1,num_classes+1),totalSoftmaxConfusionMatrix[i])
        flatSoftmaxPlots[i].set_ylim(0,100)
        flatSoftmaxPlots[i].set_ylabel("Average Softmax Response")
        flatSoftmaxPlots[i].set_xlabel("Digit")
        flatSoftmaxPlots[i].set_title(f"Softmax Response to Numerosity {i+1}")
        
        flatCorrectSoftmaxPlots[i].plot(range(1,num_classes+1),correctSoftmaxConfusionMatrix[i])
        flatCorrectSoftmaxPlots[i].plot(range(1,num_classes+1),incorrectSoftmaxConfusionMatrix[i])
        flatCorrectSoftmaxPlots[i].set_ylim(0,100)
        flatCorrectSoftmaxPlots[i].set_ylabel("Average Softmax Response")
        flatCorrectSoftmaxPlots[i].set_xlabel("Digit")
        flatCorrectSoftmaxPlots[i].set_title(f"Softmax Response to Numerosity {i+1}")
        flatCorrectSoftmaxPlots[i].legend(("Correct Classification","Incorrect Classification"))
   
    softmaxFig.savefig(os.path.join(folder,'softmax_confusion'))
    correctSoftmaxFig.savefig(os.path.join(folder,'correctvnot_softmax_confusion'))
    responseFig.savefig(os.path.join(folder,'response_confusion'))     
    plt.close('all')


def decayLR(optim, gamma):
    for g in optim.param_groups:
        g['lr']*=gamma

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


class AlexNetClassifier(nn.Module):
    def __init__(self,hidden_layer_size,num_classes):
        super(AlexNetClassifier,self).__init__()

        model = models.alexnet(pretrained=True)
        final_layer_size = 256*6*6
        self.features = model.features        
        self.classifier = nn.Sequential(
            nn.Dropout(0.9),
            nn.Linear(final_layer_size,hidden_layer_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer_size, num_classes)
        )
        
    def forward(self, x):

        x = self.features(x)
        x=torch.flatten(x,1)

        output = self.classifier(x)
        if self.training:
            return output
        return x,output

def saveAlexNetClassifier(model, path,hidden_layer_size,num_classes):
    torch.save({'model_state_dict':model.state_dict(),'hidden_layer_size':model.classifier[1].out_features, "num_classes":model.classifier[3].out_features},path)

def loadAlexNetClassifier(path):
    checkpoint=torch.load(path)
    model=AlexNetClassifier(checkpoint['hidden_layer_size'],checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model



def saveFinetune(model, path):
    torch.save({'model_state_dict':model.state_dict(),'num_hidden_layers':model.num_hidden_layers,'hidden_layer_size':model.hidden_layer_size, "num_classes":model.num_classes, "model_name":model.model_name,"dropout":model.dropout},path)

def loadFinetune(path):
    checkpoint=torch.load(path)
    model=Finetune(checkpoint.get('model_name','cornet_s'),checkpoint.get('num_hidden_layers',1),checkpoint.get('hidden_layer_size',4096),checkpoint.get('dropout',0.75),checkpoint.get('num_classes',9))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model



def initialize_model(model_name, num_classes, feature_extract,finetune=False, replace_classifier=False,num_hidden_layers=0,hidden_layer_size=0,dropout=0,use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model = None
    input_size = -1
    final_layer_size = -1


    if model_name == "resnet":
        """ Resnet18
        """
        model = models.resnet18(pretrained=use_pretrained)

        if feature_extract:    
            no_grad(model)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif replace_classifier:
            model=Finetune(model_name, hidden_layer_size,dropout, num_classes)
        elif finetune: 
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model = models.alexnet(pretrained=use_pretrained)
        if feature_extract:    
            no_grad(model)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        elif replace_classifier:
            model=Finetune(model_name, num_hidden_layers,hidden_layer_size,dropout, num_classes)
        elif finetune:
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,num_classes)

        input_size = 224


    elif model_name == "vgg":
        """ VGG11_bn
        """
        model = models.vgg11_bn(pretrained=use_pretrained)
        if feature_extract:    
            no_grad(model)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        elif replace_classifier:
            model=Finetune(model_name, num_hidden_layers,hidden_layer_size,dropout, num_classes)
        elif finetune:
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,num_classes)

        input_size = 224


    elif model_name == "densenet":
        """ Densenet
        """
        model = models.densenet121(pretrained=use_pretrained)
        if feature_extract:
            no_grad(model)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
        elif replace_classifier:
            model=Finetune(model_name, num_hidden_layers,hidden_layer_size,dropout, num_classes)
        elif finetune:    
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "cornet_s":
        model = cornet.cornet_s(pretrained=use_pretrained)
        # model = CORnet_S()
        # if use_pretrained:
        #     hash = '1d3f7974'
        #     url = f'https://s3.amazonaws.com/cornet-models/cornet_s-{hash}.pth'
        #     ckpt_data = torch.utils.model_zoo.load_url(url, map_location=None)
        #     model.load_state_dict(ckpt_data['state_dict'])

        if feature_extract:
            no_grad(model)
            num_ftrs = model.decoder[2].in_features
            model.decoder[2] = nn.Linear(num_ftrs,num_classes)
        elif replace_classifier:
            model=Finetune(model_name, num_hidden_layers,hidden_layer_size,dropout, num_classes)
        elif finetune:
            num_ftrs = model.decoder[2].in_features
            model.decoder[2] = nn.Linear(num_ftrs,num_classes)

        input_size=224

    elif model_name == "thirtycnn":
        model = ThirtyCNN(hidden_layer_size,num_classes)


    else:
        # Loading model from modeldir (model_name)
        if 'replace_classifier' in model_name:
            model = loadFinetune(os.path.join(model_name,'model.pt'))
        else:
            model = torch.load(os.path.join(model_name,'model.pt'))
        model.num_classes=num_classes

        if feature_extract:    
            no_grad(model)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

        elif replace_classifier:
            if num_hidden_layers == 1:
                model.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(model.final_layer_size,hidden_layer_size),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_layer_size, num_classes)
                )

            else:
                model.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(model.final_layer_size,hidden_layer_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_layer_size,hidden_layer_size),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_layer_size, num_classes)
                )
            model.num_hidden_layers = num_hidden_layers
            model.hidden_layer_size=hidden_layer_size
            model.dropout=dropout

        elif finetune: 
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

        input_size = 224

    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    return model, input_size, params_to_update
    

