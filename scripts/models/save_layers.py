import os
import argparse
import time
import torch
import torch.nn as nn

from . import utility_functions as utils
from .downsamplehook import DownsampleHook
from ..stimuli import data_classes

def saveLayers(model, device, data_loader, dataset_paths, hooks):
    layer_csvs = []

    model.eval()
    with torch.no_grad():
        for batch,sample in enumerate(data_loader):
            print(f"Saving batch {batch}/{len(data_loader)}")

            sample = [x.to(device) if torch.is_tensor(x) else x for x in sample]

            images = sample[0]
            stats_lists = [utils.tensorToNumpy(x) if torch.is_tensor(x) else x for x in sample[1:]]

            model(images)

            for idx,hook in enumerate(hooks):
                layer_activations = hook.output.flatten(start_dim=1)
                _, layer_size = layer_activations.size()
                layer_activations = utils.tensorToNumpy(layer_activations).tolist()

                if batch == 0:
                    csv_file = utils.createActivationCSV(dataset_paths[idx], data_loader.dataset.get_header(), layer_size)
                    layer_csvs.append(csv_file)

                csv_file = layer_csvs[idx]

                for image_stats in zip(*(stats_lists + [layer_activations])):
                    output_string = ",".join([str(x) for x in image_stats[:-1]])+','+utils.listToString(image_stats[-1])
                    utils.writeAndFlush(csv_file,output_string)

    for csv_file in layer_csvs:
        csv_file.close()


if __name__ == '__main__':

    start_time = time.time()

        #Command Line Arguments 
    parser = argparse.ArgumentParser(description='Save layer activations of a given model to given stimuli.')

    parser.add_argument('--model_name', type=str, 
                        help='neural net model to use. see scripts.models.utility_functions for the options')

    parser.add_argument('--experiment_name', type=str,
                        help='Name of experiment')
    parser.add_argument('--dataset_name', type=str,
                        help='Name of dataset to store activations for')
    parser.add_argument('--layers', nargs='+', type=str,
                        help='Names of layers to save')
    parser.add_argument('--downsample_layers', action='store_true', default=False,
                        help='Whether randomly downsample layers before storing activations')    
    parser.add_argument('--num_kept_neurons', type=int, default=4096,
                        help='Number of neurons to keep when downsampling (default: 4096)')

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
    args.dataset_type = args.dataset_name.split("_")[0]

    print('running with args:')
    print(args)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.use_cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.use_cuda else {}

    # Load model
    print("Loading model...")
    model, data_transform, input_size = utils.initializeModel(args.model_name)

    if args.multi_gpu:
        model = nn.DataParallel(model)
    model.to(device)


    # Get path to data directory
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    # Locate stimulus directory
    stim_path = os.path.join(data_path, 'stimuli',args.experiment_name, args.dataset_name,'stimuli')
    if not os.path.exists(stim_path):
        raise ValueError(f"Stimulus directory {stim_path} doesn't exist")

    # Load stimuli
    print("Loading data...")
    if args.dataset_type == 'dewind':
        data_loader = torch.utils.data.DataLoader(
            data_classes.DewindDataSet(stim_path, data_transform),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset_type == 'nasr':
        data_loader = torch.utils.data.DataLoader(
            data_classes.NasrDataSet(stim_path, data_transform),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        raise NotImplementedError(f"Dataset type {args.dataset_type} not recognized")

    print("Creating directory structure...")
    # Create directory to store layer activations
    # Create model directory
    model_dir = args.model_name

    experiment_path = os.path.join(data_path, 'models', args.experiment_name)
    # Check if it exists first since it may have been created when saving a different dataset
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    model_path = os.path.join(experiment_path, model_dir)
    # Check if it exists first since it may have been created when saving a different dataset

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # Create layer directories in model directory and register hooks
    dataset_paths=[]
    hooks = []
    for layer in args.layers:
        layer_path = os.path.join(model_path,layer)
        # Check if it exists first since it may have been created when saving a different dataset
        if not os.path.exists(layer_path):
            os.mkdir(layer_path)

        sublayer_list = layer.split('_')
        module = model
        for sublayer in sublayer_list:
            module = module._modules[sublayer]
        hook = DownsampleHook(module, args.downsample_layers, args.num_kept_neurons)
        hooks.append(hook)

        # Create dataset directory in layer directory
        dataset_path = os.path.join(layer_path, args.dataset_name)
        os.mkdir(dataset_path)
        dataset_paths.append(dataset_path)

    print("Saving layers...")
    
    saveLayers(model, device, data_loader, dataset_paths, hooks)

    print('Total Run Time:')
    print("--- %s seconds ---" % (time.time() - start_time))
