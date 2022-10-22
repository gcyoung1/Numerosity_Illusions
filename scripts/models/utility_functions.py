import os
from torchvision import models
from torchinfo import summary
from functools import reduce


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

def createActivationCSV(folder,features_size:int):
    column_list = ['numerosity','size','spacing','num_lines']
    parameters_header = listToString(column_list)
    neuron_names = [f'n{i}' for i in range(features_size)]
    activations_header = listToString(neuron_names)
    header = parameters_header + ',' + activations_header

    csv_file = open(os.path.join(folder,f'activations.csv'),'w+')
    writeAndFlush(csv_file, header)
    return csv_file

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
    
def get_layer_size_dict(model_name:str):
    model, input_size = initializeModel(model_name, False)
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
            child_name = f"{name}_{child.var_name}"
        populate_layer_size_dict(child_name, child, layer_size_dict)
        # LayerInfo objects' children attribute contains all descendents, 
        # so we skip the ones which were already added to the layer_info_dict
        # under a submodule
        num_descendents = len(child.children)
        if num_descendents > 0:
            idx += num_descendents
        idx += 1
    return None
