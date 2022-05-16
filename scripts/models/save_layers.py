import os
import argparse
import time
import sys
import torch
import torch.nn as nn
from torchvision import transforms

from . import utility_functions as utils
from .hook import Hook
from ..stimuli import data_classes

def saveLayers(model, device, data_loader, dataset_name, layer_dirs, hooks):
    layer_csvs = []

    model.eval()
    with torch.no_grad():
        for batch,(images,numerosities,sizes,spacings,line_nums) in enumerate(data_loader):
            print(f"Saving batch {batch}/len(data_loader)")

            images, numerosities,sizes,spacings,line_nums = images.to(device), numerosities.to(device),sizes.to(device), spacings.to(device), line_nums.to(device)
            numerosities = utils.tensorToNumpy(numerosities)
            sizes = utils.tensorToNumpy(sizes)
            spacings = utils.tensorToNumpy(spacings)
            line_nums = utils.tensorToNumpy(line_nums)

            model(images)

            for idx,hook in enumerate(hooks):
                layer_activations = hook.output.flatten(start_dim=1)
                _, layer_size = layer_activations.size()
                layer_activations = utils.tensorToNumpy(layer_activations).tolist()

                if batch == 0:
                    csv_file = utils.createActivationCSV(layer_dirs[idx],dataset_name,layer_size)
                    layer_csvs.append(csv_file)

                csv_file = layer_csvs[idx]


                for numerosity,size,spacing,layer_activation in zip(numerosities,sizes,spacings,layer_activations):
                    output_string = str(numerosity)+','+str(size)+','+str(spacing)+','+utils.listToString(layer_activation)
                    utils.writeAndFlush(csv_file,output_string)

    for csv_file in layer_csvs:
        csv_file.close()


if __name__ == '__main__':

    start_time = time.time()

        #Command Line Arguments 
    parser = argparse.ArgumentParser(description='Save layer activations of a given model to given stimuli.')

    parser.add_argument('--model_name', type=str, 
                        help='neural net model to use (alexnet, cornet_s)')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='If thie argument is used, initialize model at trained ImageNet weights')

    parser.add_argument('--stimulus_directory', type=str,
                        help='Name of stimulus directory in /data/stimuli to store activations for')
    parser.add_argument('--layers', nargs='+', type=str,
                        help='Names of layers to save')

    parser.add_argument('--batch_size', type=int, default=200,
                        help='input batch size for testing (default: 200)')
    parser.add_argument('--multi_gpu', action='store_true', default=False,
                        help='run model on multiple gpus')    
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 2)')
    parser.add_argument('--num_workers', type=int, default=4, metavar='W',
                        help='number of parallel batches to process. Rule of thumb 4*num_gpus (default: 4)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()

    # reconcile arguments
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()


    print('running with args:')
    print(args)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.use_cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.use_cuda else {}

    # Load model
    print("Loading model...")
    model, input_size = utils.initializeModel(args.model_name, args.pretrained)

    if args.multi_gpu:
        model = nn.DataParallel(model)
    model.to(device)


    # Get path to data directory
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    # Locate stimulus directory
    stim_path = os.path.join(data_path, 'stimuli',args.stimulus_directory,'stimuli')
    if not os.path.exists(stim_path):
        raise ValueError(f"Stimulus directory {stim_path} doesn't exist")

    # Load stimuli
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print("Loading data...")
    data_loader = torch.utils.data.DataLoader(
        data_classes.DewindDataSet(stim_path, data_transform),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    print("Creating directory structure...")
    # Create directory to store layer activations
    # Create model directory
    pretraining_status = '_pretrained' if args.pretrained else '_random'
    model_dir = args.model_name + pretraining_status
    model_path = os.path.join(data_path, 'models',model_dir)
    # Check if it exists first since another dataset may have been saved already
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # Create dataset directory in model directory
    dataset_path = os.path.join(data_path, 'models',model_dir,args.stimulus_directory)
    # Check if it exists first since another layer may have been saved already
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    os.mkdir(dataset_path)

    # Create layer directories in dataset directory and register hooks
    layer_dirs=[]
    hooks = []
    for layer in args.layers:
        layer_dir = os.path.join(dataset_path,layer)
        os.mkdir(layer_dir)
        layer_dirs.append(layer_dir)

        sublayer_list = layer.split('_')
        module = model
        for sublayer in sublayer_list:
            module = getattr(module,sublayer)
        hook = Hook(module)
        hooks.append(hook)

    print("Saving layers...")
    saveLayers(model, device, data_loader, args.stimulus_directory, layer_dirs, hooks)

    print('Total Run Time:')
    print("--- %s seconds ---" % (time.time() - start_time))
