import os
from torchvision import models
from torchvision import transforms
from torchinfo import summary
from functools import reduce
from pytorch_pretrained_vit import ViT
from .resmlp import resmlp_24
from .deit import deit_tiny_patch16_224
from .alexnet_fractaldb import alexnet_fractaldb
from .densenet import DenseNet
from .identitynet import IdentityNet


def tensorToNumpy(tensor):
    return tensor.detach().cpu().numpy()

def noGrad(model):
    for param in model.parameters():
        param.requires_grad = False

def listToString(l:list):
    return ','.join([str(x) for x in l])

def writeAndFlush(csv_file, line:str):
    csv_file.write(line + '\n')
    csv_file.flush()

def createActivationCSV(folder, column_list:list, features_size:int):
    parameters_header = listToString(column_list)
    neuron_names = [f'n{i}' for i in range(features_size)]
    activations_header = listToString(neuron_names)
    header = parameters_header + ',' + activations_header

    csv_file = open(os.path.join(folder,f'activations.csv'),'w+')
    writeAndFlush(csv_file, header)
    return csv_file

def initializeModel(model_name:str):
    if model_name == "alexnet_pretrained":
        model = models.alexnet(pretrained=True)
        input_size = 224
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    elif model_name == "alexnet_fractaldb":
        model = alexnet_fractaldb()
        input_size = 224
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


    elif model_name == "alexnet_random":
        model = models.alexnet(pretrained=False)
        input_size = 224
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    
    elif model_name == "densenet":
        model = DenseNet()
        input_size = 224
        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])


    elif model_name == "identitynet":
        model = IdentityNet()
        input_size = 224
        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    elif model_name == "vit_pretrained":
        model = ViT('B_16_imagenet1k', pretrained=True)
        input_size = 384
        data_transform = transforms.Compose([
            transforms.Resize((384, 384)), 
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])


    elif model_name == "vit_random":
        model = ViT('B_16_imagenet1k', pretrained=False)
        input_size = 384
        data_transform = transforms.Compose([
            transforms.Resize((384, 384)), 
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    elif model_name == "deit_pretrained":
        model = deit_tiny_patch16_224(pretrained=True, dataset='imagenet')
        input_size = 224
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    elif model_name == "deit_fractaldb":
        model = deit_tiny_patch16_224(pretrained=True, dataset='fractaldb')
        input_size = 224
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    elif model_name == "deit_random":
        model = deit_tiny_patch16_224(pretrained=False)
        input_size = 224
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    elif model_name == "resmlp24_pretrained":
        model = resmlp_24(pretrained=True)
        input_size = 224
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    elif model_name == "resmlp24_random":
        model = resmlp_24(pretrained=False)
        input_size = 224
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    else:
        raise ValueError("Model name not recognized")

    noGrad(model)
    return model, data_transform, input_size
    
def get_layer_size_dict(model_name:str):
    model, data_transform, input_size = initializeModel(model_name)
    model_stats = summary(model, input_size = (1,3,input_size, input_size), col_names=["output_size"])

    layer_size_dict = {}
    # Initialize name as blank since first module is whole model
    populate_layer_size_dict("", model_stats.summary_list[0], layer_size_dict)
    return layer_size_dict


def populate_layer_size_dict(name, layer_info, layer_size_dict):
    """ 
    Recursively populates the layer_size_dict with all the sublayers of the 
    passed LayerInfo object. Names of sublayers are continuations on the passed
    name, which corresponds to the name of the passed LayerInfo object. 
    E.g. for Alexnet, 
    name: name of the passed LayerInfo object's corresponding layer
    layer_info: LayerInfo object to populate the layer_size_dict with
    layer_size_dict: dict whose keys are layer names and values are number of neurons
    """

    if layer_info.is_leaf_layer:
        # Collapse multi-dimensional output into number of neurons
        num_neurons = reduce((lambda x, y: x * y), layer_info.output_size)
        layer_size_dict[name] = num_neurons
        return None
    idx = 0
    while idx < len(layer_info.children):
        child  = layer_info.children[idx]
        if len(name) == 0:
            child_name = child.var_name
        else:
            child_name = f"{name}__{child.var_name}"
        populate_layer_size_dict(child_name, child, layer_size_dict)
        # LayerInfo objects' children attribute contains all descendents, 
        # so we skip the ones which were already added to the layer_info_dict
        # under a submodule
        num_descendents = len(child.children)
        if num_descendents > 0:
            idx += num_descendents
        idx += 1
    return None
