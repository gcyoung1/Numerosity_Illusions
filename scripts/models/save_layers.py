import os
import argparse
import time
import torch
import torch.nn as nn
from torchvision import transforms

from . import utility_functions as utils
from .hook import Hook
from ..stimuli import data_classes

def saveLayers(model, device, data_loader, dataset_paths, hooks):
    layer_csvs = []

    model.eval()
    with torch.no_grad():
        for batch,(images,numerosities,sizes,spacings,num_lines_list) in enumerate(data_loader):
            print(f"Saving batch {batch}/len(data_loader)")

            images, numerosities,sizes,spacings,line_nums = images.to(device), numerosities.to(device),sizes.to(device), spacings.to(device), num_lines_list.to(device)
            numerosities = utils.tensorToNumpy(numerosities)
            sizes = utils.tensorToNumpy(sizes)
            spacings = utils.tensorToNumpy(spacings)
            num_lines_list = utils.tensorToNumpy(num_lines_list)

            model(images)

            for idx,hook in enumerate(hooks):
                layer_activations = hook.output.flatten(start_dim=1)
                _, layer_size = layer_activations.size()
                layer_activations = utils.tensorToNumpy(layer_activations).tolist()

                if batch == 0:
                    csv_file = utils.createActivationCSV(dataset_paths[idx],layer_size)
                    layer_csvs.append(csv_file)

                csv_file = layer_csvs[idx]

                for numerosity,size,spacing,num_lines,layer_activation in zip(numerosities,sizes,spacings,num_lines_list,layer_activations):
                    output_string = str(numerosity)+','+str(size)+','+str(spacing)+','+str(num_lines)+','+utils.listToString(layer_activation)
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
    # Check if it exists first since it may have been created already
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # Create layer directories in model directory and register hooks
    dataset_paths=[]
    hooks = []
    for layer in args.layers:
        layer_path = os.path.join(data_path, 'models',model_dir,layer)
        # Check if it exists first since it may have been created already
        if not os.path.exists(layer_path):
            os.mkdir(layer_path)

        sublayer_list = layer.split('_')
        module = model
        for sublayer in sublayer_list:
            module = getattr(module,sublayer)
        hook = Hook(module)
        hooks.append(hook)

        # Create dataset directory in layer directory
        dataset_path = os.path.join(layer_path, args.stimulus_directory)
        os.mkdir(dataset_path)
        dataset_paths.append(dataset_path)

    print("Saving layers...")
    
    #saveLayers(model, device, data_loader, dataset_paths, hooks)

    print('Total Run Time:')
    print("--- %s seconds ---" % (time.time() - start_time))
