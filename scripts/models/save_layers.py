from __future__ import print_function
import os
import argparse
import time
import torch
import torch.nn as nn
import sys
sys.path.insert(0,'../data_prep/data_loading')
import data_classes
from torchvision import datasets, models, transforms
from utility_functions import listToString,createActivationCSV,Hook
sys.path.append('../models')

def saveDewindLayer(model, device, data_loader, dataset_name):
    model.eval()
    with torch.no_grad():
        for batch,(images,numerosities,sizes,spacings,line_nums) in enumerate(data_loader):

            images, numerosities,sizes,spacings,line_nums = images.to(device), numerosities.to(device),sizes.to(device), spacings.to(device), line_nums.to(device)
            model(images)

            if batch == 0:
                layer_csvs = []
            for idx,hook in enumerate(args.hooks):
                layerActivations = hook.output
                layerActivations = layerActivations.cpu().numpy()
                layerActivations = layerActivations.reshape(layerActivations.shape[0],-1).tolist()

                if batch == 0:
                    layer_size = len(layerActivations[0])
                    csvFile = createActivationCSV(args.layerdirs[idx],dataset_name,layer_size)
                    layer_csvs.append(csvFile)

                csvFile = layer_csvs[idx]

                numerosities = numerosities.cpu().numpy()
                sizes = sizes.cpu().numpy()
                spacings = spacings.cpu().numpy()
                line_nums = line_nums.cpu().numpy()

x                for numerosity,size,spacing,activation in zip(numerosities,sizes,spacings,layerActivations):
                    output_string = str(numerosity)+','+str(size)+','+str(spacing)+','+listToString(activation)
                    csvFile.write(output_string+'\n')
                    csvFile.flush()

    for csvFile in layer_csvs:
        csvFile.close()


def main(args):

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.use_cuda else "cpu")

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.use_cuda else {}

    #load model
    print("Loading model from input_data")
    model = torch.load(os.path.join(args.epoch_folder,'model.pt'))

    args.hooks = []
    for layer in args.layers:
        sublayer_list = layer.split('_')
        hook = model
        for sublayer in sublayer_list:
            hook = getattr(hook,sublayer)
        hook = Hook(hook)
        args.hooks.append(hook)

    if args.multi_gpu:
        model = nn.DataParallel(model)
    model.to(device)

    input_size=224
    #data loading
    data_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if args.enumeration_data:
        enumeration_test_loader = torch.utils.data.DataLoader(
            data_classes.CountingDotsDataSet(root_dir='../data/stimuli/%s'%args.enumeration_data, train=False,transform=data_transform),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
        saveLayer(model, device, enumeration_test_loader,args.num_classes,args.enumeration_data)
    if args.symbol_data:
        symbol_test_loader = torch.utils.data.DataLoader(
            data_classes.SymbolDataSet(root_dir='../data/stimuli/%s'%args.symbol_data, train=False,transform=data_transform),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
        saveLayer(model, device, symbol_test_loader,args.num_classes,args.symbol_data)
    if args.dewind_data:
        dewind_test_loader = torch.utils.data.DataLoader(
            data_classes.DewindDataSet(root_dir='../data/stimuli/%s'%args.dewind_data, train=False),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
        saveDewindLayer(model, device, dewind_test_loader,args.num_classes,args.dewind_data)
    



if __name__ == '__main__':

    start_time = time.time()

        #Command Line Arguments 
    parser = argparse.ArgumentParser(description='Save layer activations of a given model to given stimuli.')

    parser.add_argument('--model', type=str, 
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
    print('running with args:')
    print(args)

    # Create model folder
    pretraining_status = args.pretrained ? '_pretrained_' : '_random_'
    dataset_name = args.model_name + pretraining_status + time.strftime('%m-%d-%Y:%H_%M')
    outputdir = os.path.join('../../data/models',dataset_name)
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    # Locate stimulus folder
    stim_dir = os.path.join('../../data/stimuli',args.stimulus_directory,'stimuli')
    if not os.path.exists(outputdir):
        raise ValueError(f"Stimulus directory {stim_dir} doesn't exist")

    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('using cuda')

    for subfolder in sorted(os.listdir(path)):
        args.epoch_folder = os.path.join(path,subfolder)        
        if os.path.isdir(args.epoch_folder) and 'epoch' in args.epoch_folder:
            epoch = args.epoch_folder.split('epoch')[-1]
            print(f"Epoch {epoch}")

            args.layerdirs=[]
            for layer in args.layers:
#                shutil.rmtree(os.path.join(args.epoch_folder,layer))
                if not os.path.exists(os.path.join(args.epoch_folder,layer)):
                    os.mkdir(os.path.join(args.epoch_folder,layer))
                args.layerdirs.append(os.path.join(args.epoch_folder,layer))
            main(args)
            

    print('Total Run Time:')
    print("--- %s seconds ---" % (time.time() - start_time))