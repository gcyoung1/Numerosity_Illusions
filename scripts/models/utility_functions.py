import os
from torchvision import models
from torchvision import transforms
from functools import reduce
from pytorch_pretrained_vit import ViT
import torch

from .resmlp import resmlp_24
from .deit import deit_tiny_patch16_224
from .alexnet_fractaldb import alexnet_fractaldb
from .densenet import DenseNet
from .identitynet import IdentityNet
from .sizehook import SizeHook


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

def createActivationCSV(folder, column_list:list, indices):
    parameters_header = listToString(column_list)
    neuron_names = [f'n{i}' for i in indices]
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

    sizehook_dict = {}
    # Initialize name as blank
    populate_sizehook_dict("", model, sizehook_dict)

    # Pass image through network and record layer sizes with SizeHooks
    image_size = (1,3,input_size, input_size)
    dummy_image = torch.zeros(image_size)
    model(dummy_image)

    # Extract layer sizes from SizeHooks
    layer_size_dict = {layer: sizehook.output for layer, sizehook in sizehook_dict.items()}
    return layer_size_dict


def populate_sizehook_dict(surname, module, sizehook_dict):
    """ 
    Recursively populates the sizehook_dict with SizeHooks for each
    base layer in the module.
    surname: name of the submodules above this module, hierarchically
    modules: the module to populate the dict with
    sizehook_dict: dict whose keys are layer names and values are SizeHooks
    """
    submodules = module._modules
    if len(submodules) == 0:
        # Base case: if you have no submodules, create a hook and add yourself to the dict
        hook = SizeHook(module)
        sizehook_dict[surname] = hook
        return None

    for submodule_name, submodule in submodules.items():
        # Recursive case: if you have submodules, pass down your name
        new_surname = f"{surname}_{submodule_name}" if surname != "" else submodule_name
        populate_sizehook_dict(new_surname, submodule, sizehook_dict)
    return None
