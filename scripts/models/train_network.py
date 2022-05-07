from __future__ import print_function
import os
import argparse
import time
import torch
#torch.set_printoptions(threshold=5000)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.insert(0,'./data_generation')
import data_classes
from torchvision import datasets, models, transforms
from utility_functions import initialize_model,decayLR,createAccuracyCSV, createActivationCSV,Hook,listToString,saveFinetune,getClassifierParams,makeConfusionPlots,set_bn_eval
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import pickle


def train(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    if args.fc:
        model.apply(set_bn_eval)
    epochLoss=0
    allTruths = np.array([])
    allPredictions = np.array([])

    batch_accuracies = []
    for batch_idx, (data,numerosities) in enumerate(train_loader):

        data, numerosities = data.to(device), numerosities.to(device)
        target = numerosities - args.lowest_number
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()
        epochLoss+=loss.item()

        softmaxes = F.softmax(outputs,dim=1).cpu().detach().numpy()
        predictions = np.argmax(softmaxes, axis=1)
        groundTruths = target.cpu().detach().numpy()
        allPredictions = np.concatenate((allPredictions,predictions))
        allTruths = np.concatenate((allTruths,groundTruths))

        if not args.no_train_log:        
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))              
                temp_classification_report_dict = classification_report(allTruths,allPredictions,output_dict=True)
                batch_accuracies.append(temp_classification_report_dict['accuracy']*100)

    classification_report_dict = classification_report(allTruths,allPredictions,output_dict=True)
    print(f"Accuracy across batch: {batch_accuracies}")
    return epochLoss/len(train_loader.dataset), classification_report_dict['accuracy']*100


def testComparison(args, model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, numerosities, img_name, tot, num_blue, num_red, ratio, index in test_loader:
            data, numerosities = data.to(device), numerosities.to(device)
            target = numerosities - args.lowest_number
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if args.save_outputs:
                for i in range(len(target)):
                    args.testing_log.write('%s_%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n'%(str(epoch),str(i),str(int(pred[i])),str(int(target[i])),str(float(output[i][0])),
                                        str(float(output[i][1])),str(int(tot[i])),str(int(num_blue[i])),str(int(num_red[i])),str(float(ratio[i])),str(int(index[i])),img_name[i]))
                    args.testing_log.flush()

    if args.save_outputs:
        torch.save(output, os.path.join('../data/outputs',args.outputdir,'%s_output.pt'%str(epoch)))
        torch.save(target, os.path.join('../data/outputs',args.outputdir,'%s_target.pt'%str(epoch)))

    if not args.no_test_log:
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    return test_loss

def testClassification(args, model, device, test_loader, criterion, epoch,dataset_name):
    #pdb.set_trace()
    model.eval()
    test_loss = 0
    correct = 0
    allTruths = np.array([])
    allPredictions = np.array([])

    if args.confusion_matrices:
        confusionMatrix = np.zeros((args.num_classes,args.num_classes))
        correctSoftmaxConfusionMatrix = np.zeros((args.num_classes,args.num_classes))
        incorrectSoftmaxConfusionMatrix = np.zeros((args.num_classes,args.num_classes))
        correctCounts = 0.0000000001*np.ones(args.num_classes)
        incorrectCounts = 0.0000000001*np.ones(args.num_classes)
        os.mkdir(os.path.join(args.epochFolder,f'{dataset_name}_confusion'))
        confusionFolder = os.path.join(args.epochFolder,f'{dataset_name}_confusion')

    with torch.no_grad():
        for batch,(data,numerosities,conditions) in enumerate(test_loader):
            data, numerosities = data.to(device), numerosities.to(device)
            target = numerosities - args.lowest_number
            numerosities = numerosities.cpu().numpy()
            outputs = model(data)

            test_loss += criterion(outputs, target).item()

            softmaxes = F.softmax(outputs,dim=1).cpu().numpy()
            predictions = np.argmax(softmaxes, axis=1)
            groundTruths = target.cpu().numpy()
            correct += np.sum(predictions==groundTruths)
            allPredictions = np.concatenate((allPredictions,predictions))
            allTruths = np.concatenate((allTruths,groundTruths))

            if args.confusion_matrices:

                for truth,prediction,softmax in zip(groundTruths,predictions,softmaxes):
                    if truth==prediction:
                        correctCounts[truth]+=1
                        correctSoftmaxConfusionMatrix[truth] += softmax
                    else:
                        incorrectCounts[truth]+=1
                        incorrectSoftmaxConfusionMatrix[truth] += softmax
                    confusionMatrix[truth,prediction]+=1



            if args.save_outputs:
                torch.save(output, os.path.join('../data/outputs',args.outputdir,'%s_output.pt'%str(epoch)))
                torch.save(target, os.path.join('../data/outputs',args.outputdir,'%s_target.pt'%str(epoch)))

            if len(args.save_layers) > 0:


                if batch == 0:
                    csvList = []
                for idx,hook in enumerate(args.hooks):
                    layerActivations = hook.output
                    layerActivations = layerActivations.cpu().numpy()
                    layerActivations = layerActivations.reshape(layerActivations.shape[0],-1).tolist()

                    if batch == 0:
                        layer_size = len(layerActivations[0])
                        csvFile = createActivationCSV(args.layerdirs[idx],dataset_name,layer_size)
                        csvList.append(csvFile)
                    
                    csvFile = csvList[idx]

                    for numerosity,condition,activation in zip(numerosities,conditions,layerActivations):
                        output_string = str(numerosity)+','+str(condition)+','+listToString(activation)
                        csvFile.write(output_string+'\n')
                        csvFile.flush()
                



    if not args.no_test_log:
        test_loss /= len(test_loader.dataset)
        print('epoch: ' + str(epoch))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    if args.save_outputs:
        output_string = indiv_acc_dict[0][3]
        for i in range(args.num_classes):
            output_string += ','+indiv_acc_dict[i][3]
        args.testing_log.write(output_string+'\n')
        args.testing_log.flush()

    if len(args.save_layers) > 0:
        for csvFile in csvList:
            csvFile.close()

    if args.confusion_matrices:
            responseConfusionMatrix = 100*np.divide(confusionMatrix.T,correctCounts+incorrectCounts).T
            totalSoftmaxConfusionMatrix = 100*np.divide((correctSoftmaxConfusionMatrix+incorrectSoftmaxConfusionMatrix).T,correctCounts+incorrectCounts).T
            correctSoftmaxConfusionMatrix = 100*np.divide(correctSoftmaxConfusionMatrix.T,correctCounts).T
            incorrectSoftmaxConfusionMatrix = 100*np.divide(incorrectSoftmaxConfusionMatrix.T,incorrectCounts).T

            np.save(os.path.join(confusionFolder,'response_confusion_matrix'), responseConfusionMatrix)
            np.save(os.path.join(confusionFolder,'correct_softmax_confusion_matrix'), correctSoftmaxConfusionMatrix)
            np.save(os.path.join(confusionFolder,'incorrect_softmax_confusion_matrix'), incorrectSoftmaxConfusionMatrix)
            np.save(os.path.join(confusionFolder,'total_softmax_confusion_matrix'), totalSoftmaxConfusionMatrix)

            makeConfusionPlots(confusionFolder,responseConfusionMatrix,correctSoftmaxConfusionMatrix,incorrectSoftmaxConfusionMatrix,totalSoftmaxConfusionMatrix)

    classification_report_dict = classification_report(allTruths,allPredictions,output_dict=True)
    f = open(os.path.join(args.epochFolder,f'{dataset_name}_classification_report_dict.pkl'), 'wb')
    pickle.dump(classification_report_dict, f)
    f.close()

    return test_loss,classification_report_dict['accuracy']*100








def testDewind(args, model, device, test_loader, criterion, epoch,dataset_name):
    #pdb.set_trace()
    model.eval()
    test_loss = 0
    correct = 0
    allTruths = np.array([])
    allPredictions = np.array([])

    if args.confusion_matrices:
        confusionMatrix = np.zeros((args.num_classes,args.num_classes))
        correctSoftmaxConfusionMatrix = np.zeros((args.num_classes,args.num_classes))
        incorrectSoftmaxConfusionMatrix = np.zeros((args.num_classes,args.num_classes))
        correctCounts = 0.0000000001*np.ones(args.num_classes)
        incorrectCounts = 0.0000000001*np.ones(args.num_classes)
        os.mkdir(os.path.join(args.epochFolder,f'{dataset_name}_confusion'))
        confusionFolder = os.path.join(args.epochFolder,f'{dataset_name}_confusion')

    with torch.no_grad():
        for batch,(data,numerosities,square,bounding) in enumerate(test_loader):
            data, numerosities,square,bounding = data.to(device), numerosities.to(device), square.to(device),bounding.to(device)
            target = numerosities - args.lowest_number
            numerosities = numerosities.cpu().numpy()
            outputs = model(data)

            test_loss += criterion(outputs, target).item()

            softmaxes = F.softmax(outputs,dim=1).cpu().numpy()
            predictions = np.argmax(softmaxes, axis=1)
            groundTruths = target.cpu().numpy()
            correct += np.sum(predictions==groundTruths)
            allPredictions = np.concatenate((allPredictions,predictions))
            allTruths = np.concatenate((allTruths,groundTruths))

            if args.confusion_matrices:

                for truth,prediction,softmax in zip(groundTruths,predictions,softmaxes):
                    if truth==prediction:
                        correctCounts[truth]+=1
                        correctSoftmaxConfusionMatrix[truth] += softmax
                    else:
                        incorrectCounts[truth]+=1
                        incorrectSoftmaxConfusionMatrix[truth] += softmax
                    confusionMatrix[truth,prediction]+=1



            if args.save_outputs:
                torch.save(output, os.path.join('../data/outputs',args.outputdir,'%s_output.pt'%str(epoch)))
                torch.save(target, os.path.join('../data/outputs',args.outputdir,'%s_target.pt'%str(epoch)))

            if len(args.save_layers) > 0:

                if batch == 0:
                    csvList = []
                for idx,hook in enumerate(args.hooks):
                    layerActivations = hook.output
                    layerActivations = layerActivations.cpu().numpy()
                    layerActivations = layerActivations.reshape(layerActivations.shape[0],-1).tolist()

                    if batch == 0:
                        layer_size = len(layerActivations[0])
                        csvFile = createActivationCSV(args.layerdirs[idx],dataset_name,layer_size)
                        csvList.append(csvFile)
                    
                    csvFile = csvList[idx]
                    numerosities = numerosities.cpu().numpy()
                    square_sides = square.cpu().numpy()
                    bounding_sides = bounding.cpu().numpy()

                    for numerosity,square_side,bounding_side,activation in zip(numerosities,square_sides,bounding_sides,layerActivations):
                        output_string = str(numerosity)+','+str(square_side)+','+str(bounding_side)+','+listToString(activation)
                        csvFile.write(output_string+'\n')
                        csvFile.flush()
                



    if not args.no_test_log:
        test_loss /= len(test_loader.dataset)
        print('epoch: ' + str(epoch))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    if args.save_outputs:
        output_string = indiv_acc_dict[0][3]
        for i in range(args.num_classes):
            output_string += ','+indiv_acc_dict[i][3]
        args.testing_log.write(output_string+'\n')
        args.testing_log.flush()

    if len(args.save_layers) > 0:
        for csvFile in csvList:
            csvFile.close()

    if args.confusion_matrices:
            responseConfusionMatrix = 100*np.divide(confusionMatrix.T,correctCounts+incorrectCounts).T
            totalSoftmaxConfusionMatrix = 100*np.divide((correctSoftmaxConfusionMatrix+incorrectSoftmaxConfusionMatrix).T,correctCounts+incorrectCounts).T
            correctSoftmaxConfusionMatrix = 100*np.divide(correctSoftmaxConfusionMatrix.T,correctCounts).T
            incorrectSoftmaxConfusionMatrix = 100*np.divide(incorrectSoftmaxConfusionMatrix.T,incorrectCounts).T

            np.save(os.path.join(confusionFolder,'response_confusion_matrix'), responseConfusionMatrix)
            np.save(os.path.join(confusionFolder,'correct_softmax_confusion_matrix'), correctSoftmaxConfusionMatrix)
            np.save(os.path.join(confusionFolder,'incorrect_softmax_confusion_matrix'), incorrectSoftmaxConfusionMatrix)
            np.save(os.path.join(confusionFolder,'total_softmax_confusion_matrix'), totalSoftmaxConfusionMatrix)

            makeConfusionPlots(confusionFolder,responseConfusionMatrix,correctSoftmaxConfusionMatrix,incorrectSoftmaxConfusionMatrix,totalSoftmaxConfusionMatrix)

    classification_report_dict = classification_report(allTruths,allPredictions,output_dict=True)
    f = open(os.path.join(args.epochFolder,f'{dataset_name}_classification_report_dict.pkl'), 'wb')
    pickle.dump(classification_report_dict, f)
    f.close()

    return test_loss,classification_report_dict['accuracy']*100




def testEstimation(args, model, device, test_loader, criterion, epoch):
    #pdb.set_trace()
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data,numerosities in test_loader:
            data, numerosities = data.to(device), numerosities.to(device)
            target = numerosities - args.lowest_number
            numerosities = numerosities.cpu().numpy()
            output = model(data)
            test_loss += criterion(output.squeeze(), target.float()).item()

            if args.save_outputs:
                torch.save(output, os.path.join('../data/outputs',args.outputdir,'%s_output.pt'%str(epoch)))
                torch.save(target, os.path.join('../data/outputs',args.outputdir,'%s_target.pt'%str(epoch)))
                    

        if not args.no_test_log:
            test_loss /= len(test_loader.dataset)
            print('epoch: ' + str(epoch))
            print('\nTest set: Average loss: {:.4f}'.format(test_loss))

    return test_loss

def testSolitaire(args,model, device, solitaire_loader):
    model.eval()
    solitaireCSV = createActivationCSV(args.solitairedir,'solitiare',0)
    with torch.no_grad():
        for red_images,blue_images,num_rows,group_sizes,togethers in solitaire_loader:
            red_images,blue_images = red_images.to(device),blue_images.to(device)
            red_solid_estimates = model(red_images)
            blue_solid_estimates = model(blue_images)
            ratios = blue_solid_estimates/red_solid_estimates
            ratios = ratios.cpu().numpy()
            group_sizes = group_sizes.cpu().numpy()
            num_rows = num_rows.cpu().numpy()

            for redimage,num_row,group_size,together,ratio in zip(red_images,num_rows,group_sizes,togethers,ratios):
                output_string = str(num_row)+','+str(group_size)+','+str(together)+','+str(ratio[0])
                solitaireCSV.write(output_string+'\n')
                solitaireCSV.flush()



def main(args):
    if not args.randomize:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    if args.feature_extract:
        training_type='feature_extract'
    elif args.replace_classifier:
        training_type="replace_classifier"
    elif args.finetune:
        training_type = "finetune"
    else:
        training_type = "as_is"

    args.outputdir = '%s%s%s%s%s%s_%s_%s_%s'%(args.comparison_data.replace('/','_'),args.estimation_data.replace('/','_'),args.solitaire_data.replace('/','_'),args.enumeration_data.replace('/','_'),args.symbol_data.replace('/','_'),args.dewind_data.replace('/','_'),training_type,args.model,time.strftime('%m-%d-%Y:%H_%M'))        
    args.starting_epoch = 0

    #load model
    if args.modeldir:
        args.modeldir = os.path.join('../data/outputs',args.modeldir)
        print(f"Loading model from {args.modeldir}")
        model, input_size, params_to_update = initialize_model(args.modeldir, args.num_classes, args.feature_extract, args.finetune,  args.replace_classifier,args.num_hidden_layers,args.hidden_layer_size,args.dropout,use_pretrained=args.pretrained)

        if args.continue_training:
            split_on_epoch = args.modeldir.split('epoch')
            args.outputdir = split_on_epoch[0]
            args.starting_epoch = int(split_on_epoch[1])
        else: 
            if not os.path.exists(os.path.join(args.modeldir,args.outputdir)):
                os.mkdir(os.path.join(args.modeldir,args.outputdir))
            args.outputdir = os.path.join(args.modeldir,args.outputdir)

    else:
        model, input_size, params_to_update = initialize_model(args.model, args.num_classes, args.feature_extract,  args.finetune, args.replace_classifier,args.num_hidden_layers,args.hidden_layer_size,args.dropout,use_pretrained=args.pretrained)

        if not os.path.exists(os.path.join('../data/outputs',args.outputdir)):
            os.mkdir(os.path.join('../data/outputs',args.outputdir))
        args.outputdir=os.path.join('../data/outputs',args.outputdir)

    args_file = open(os.path.join(args.outputdir,'args.txt'),'a')
    args_file.write(str(args))
    args_file.close()


    #load forward hooks
    if args.save_layers:
        args.hooks = []
    for layer in args.save_layers:
        sublayer_list = layer.split('_')
        hook = model
        for sublayer in sublayer_list:
            hook = getattr(hook,sublayer)
        hook = Hook(hook)
        args.hooks.append(hook)

    model.to(device)
    if args.multi_gpu:
        model = nn.DataParallel(model)

    #data loading
    data_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if args.symbol_data:
        if args.enumeration_data:
            train_loader = torch.utils.data.DataLoader(
                data_classes.EnumerationAndSymbolsDataSet(enumeration_root='../data/stimuli/'+args.enumeration_data,symbol_root='../data/stimuli/'+args.symbol_data, train=True,transform=data_transform),
                batch_size=args.batch_size, shuffle=True, **kwargs)

            enumeration_test_loader = torch.utils.data.DataLoader(
                data_classes.CountingDotsDataSet(root_dir='../data/stimuli/%s'%args.enumeration_data, train=False,transform=data_transform),
                batch_size=args.test_batch_size, shuffle=False, **kwargs)

            symbol_test_loader = torch.utils.data.DataLoader(
                data_classes.SymbolDataSet(root_dir='../data/stimuli/'+args.symbol_data,train=False,transform=data_transform),
                batch_size=args.test_batch_size, shuffle=False, **kwargs)
        else:
            train_loader = torch.utils.data.DataLoader(
                data_classes.SymbolDataSet(root_dir='../data/stimuli/'+args.symbol_data, train=True,transform=data_transform),
                batch_size=args.batch_size, shuffle=True, **kwargs)
            
            symbol_test_loader = torch.utils.data.DataLoader(
                data_classes.SymbolDataSet(root_dir='../data/stimuli/'+args.symbol_data,train=False,transform=data_transform),
                batch_size=args.test_batch_size, shuffle=False, **kwargs)


    elif args.enumeration_data:
        train_loader = torch.utils.data.DataLoader(
            data_classes.CountingDotsDataSet(root_dir='../data/stimuli/'+args.enumeration_data, train=True,transform=data_transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        enumeration_test_loader = torch.utils.data.DataLoader(
            data_classes.CountingDotsDataSet(root_dir='../data/stimuli/%s'%args.enumeration_data, train=False,transform=data_transform),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

    elif args.estimation_data:
        train_loader = torch.utils.data.DataLoader(
            data_classes.EstimatingDotsDataSet(root_dir='../data/stimuli/'+args.estimation_data, train=True,transform=data_transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        estimation_test_loader = torch.utils.data.DataLoader(
            data_classes.EstimatingDotsDataSet(root_dir='../data/stimuli/%s'%args.estimation_data, train=False,transform=data_transform),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
        
        if args.solitaire_data:
            solitaire_test_loader = torch.utils.data.DataLoader(
                data_classes.SolitaireDataSet(root_dir='../data/stimuli/'+args.solitaire_data,transform=data_transform),
                batch_size=args.batch_size, shuffle=False, **kwargs)
    
    elif args.comparison_data:
        train_loader = torch.utils.data.DataLoader(
            data_classes.ComparisonDotsDataSet(root_dir='../data/stimuli/'+args.comparison_data, train=True,transform=data_transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        comparison_test_loader = torch.utils.data.DataLoader(
            data_classes.ComparisonDotsDataSet(root_dir='../data/stimuli/%s'%args.comparison_data, train=False,transform=data_transform),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

    elif args.dewind_data:
        #train_data = data_classes.DewindDataSet(root_dir='../data/stimuli/'+args.dewind_data, train=True)
        train_loader = torch.utils.data.DataLoader(
            data_classes.DewindDataSet(root_dir='../data/stimuli/'+args.dewind_data, train=True),
            batch_size=args.batch_size, shuffle=True, **kwargs)            

        dewind_test_loader = torch.utils.data.DataLoader(
            data_classes.DewindDataSet(root_dir='../data/stimuli/%s'%args.dewind_data, train=False),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
        
        # num_partitions = int(args.epochs/args.subset_interval)
        # indices = list(range(len(train_data)))
        # np.random.shuffle(indices)
        # partitions = np.array_split(indices,num_partitions)

  
    #Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(params_to_update, lr=args.lr)
        if args.fc_epochs:
            classifier_params = getClassifierParams(model)
            fc_optimizer = optim.Adam(classifier_params,lr=args.fc_lr)

    else:
        optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=args.momentum)
        if args.fc_epochs:
            classifier_params = getClassifierParams(model)
            fc_optimizer = optim.SGD(classifier_params,lr=args.fc_lr,momentum=args.momentum)

    #loss
    if args.estimation_data:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()



    if args.save_outputs:
        if args.comparison_data:
            args.testing_log = open(os.path.join(args.outputdir,'testing_log.csv'),'a')
            args.testing_log.write('trial,prediction,target,net_output1,net_output2,tot,num_blue,num_red,ratio,index,img_name')
            args.testing_log.write('\n')
            args.testing_log.flush()
        elif args.enumeration_data:
            args.testing_log = open(os.path.join(args.outputdir,'testing_log.csv'),'a')#Need to differentiate symbol and enumeration
            for i in range(1,args.num_classes+1):
                args.testing_log.write('%s acc,%s guessed,%s F1'%(str(i),str(i),str(i)))
                args.testing_log.write('\n')
                args.testing_log.flush()


    if args.continue_training:
        testEpochs = list(np.load(os.path.join(args.outputdir,'testEpochs.npy')))
        trainLossHistory=list(np.load(os.path.join(args.outputdir,'trainLossHistory.npy')))
        trainAccuracyHistory=list(np.load(os.path.join(args.outputdir,'trainAccuracyHistory.npy')))

        testLossHistory=list(np.load(os.path.join(args.outputdir,'testLossHistory.npy')))
        testAccuracyHistory=list(np.load(os.path.join(args.outputdir,'testAccuracyHistory.npy')))

        if args.enumeration_data and args.symbol_data:
            symbol_testAccuracyHistory=list(np.load(os.path.join(args.outputdir,'symbol_testAccuracyHistory.npy')))
            symbol_testLossHistory=list(np.load(os.path.join(args.outputdir,'symbol_testLossHistory.npy')))
        
    else:
        testEpochs = []      
        trainLossHistory=[]
        trainAccuracyHistory=[]

        testAccuracyHistory=[]
        testLossHistory=[]
        if args.enumeration_data and args.symbol_data:
            symbol_testAccuracyHistory=[]
            symbol_testLossHistory=[]

    
    for epoch in args.starting_epoch + np.arange(0,args.fc_epochs+args.epochs + 1):
        if (not(epoch == args.starting_epoch and args.starting_epoch > 0)) and (epoch%args.test_interval == 0 or epoch < args.initial_test_period or epoch in (args.starting_epoch + np.arange(args.fc_epochs,args.fc_epochs+args.initial_test_period))):
            print(f"Testing Epoch {epoch}")
            testEpochs.append(epoch)
            os.mkdir(os.path.join(args.outputdir,f'epoch{epoch:03}'))#zero padding epoch
            args.epochFolder = os.path.join(args.outputdir,f'epoch{epoch:03}')
            if args.save_model:
                if args.modeldir or args.replace_classifier: #imperfect bc it's still possible to load a non-finetune model from a modeldir
                    saveFinetune(model,os.path.join(args.epochFolder,'model.pt'))
                else:
                    torch.save(model,os.path.join(args.epochFolder,'model.pt'))
            
            if args.save_layers:
                args.layerdirs=[]
                for layer in args.save_layers:
                    os.mkdir(os.path.join(args.epochFolder,layer))
                    args.layerdirs.append(os.path.join(args.epochFolder,layer))


            if args.comparison_data:
                testLoss,testAccuracy=testComparison(args, model, device, comparison_test_loader, criterion, epoch)
            elif args.enumeration_data:
                testLoss,testAccuracy=testClassification(args, model, device, enumeration_test_loader, criterion, epoch,args.enumeration_data)
                if args.symbol_data:
                    symbol_testLoss,symbol_testAccuracy=testClassification(args, model, device, symbol_test_loader, criterion, epoch,args.symbol_data)
                    symbol_testLossHistory.append(symbol_testLoss)
                    symbol_testAccuracyHistory.append(symbol_testAccuracy)

            elif args.symbol_data:
                testLoss,testAccuracy=testClassification(args, model, device, symbol_test_loader, criterion, epoch,args.symbol_data)
            elif args.estimation_data:
                testLoss,testAccuracy=testEstimation(args, model, device, estimation_test_loader, criterion, epoch)
            elif args.solitaire_data:
                testSolitaire(args,model, device, solitaire_loader)
            elif args.dewind_data:
                testLoss,testAccuracy=testDewind(args, model, device, dewind_test_loader, criterion, epoch,args.dewind_data)


            testLossHistory.append(testLoss)
            testAccuracyHistory.append(testAccuracy)

        if epoch < (args.starting_epoch + args.epochs+args.fc_epochs):
            if epoch < args.fc_epochs:
                args.fc=True
                trainLoss,trainAccuracy = train(args, model, device, train_loader, fc_optimizer, criterion, epoch)
                trainLossHistory.append(trainLoss)          
                trainAccuracyHistory.append(trainAccuracy)
                if args.fc_lr_decay < 1 and (epoch+1)%args.fc_lr_decay_interval==0:
                    decayLR(fc_optimizer,args.fc_lr_decay)
            else:
                args.fc=False
                trainLoss,trainAccuracy = train(args, model, device, train_loader, optimizer, criterion, epoch)
                trainLossHistory.append(trainLoss)          
                trainAccuracyHistory.append(trainAccuracy)
                if args.lr_decay < 1 and (epoch+1)%args.lr_decay_interval==0:
                    decayLR(optimizer,args.lr_decay)


    plt.plot(range(args.starting_epoch + args.fc_epochs+args.epochs),trainLossHistory)
    plt.plot(testEpochs,testLossHistory)
    if args.fc_epochs:#partition fc and full network training
        plt.vlines(args.fc_epochs,0,max([max(testLossHistory),max(trainLossHistory)]),linestyles='dashed')
    plt.ylabel("Average Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train Loss","Test Loss"])
    if args.enumeration_data and args.symbol_data:
        plt.plot(testEpochs,symbol_testLossHistory)
        plt.legend(["Train Loss","Enumeration Test Loss", "Symbol Test Loss"])
    plt.savefig(os.path.join(args.outputdir,'lossHistory'))
    plt.close()

    plt.plot(range(args.starting_epoch + args.fc_epochs+args.epochs),trainAccuracyHistory)
    plt.plot(testEpochs,testAccuracyHistory)
    if args.fc_epochs:
        plt.vlines(args.fc_epochs,0,100 ,linestyles='dashed')
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train Accuracy","Test Accuracy"])
    if args.enumeration_data and args.symbol_data:
        plt.plot(testEpochs,symbol_testAccuracyHistory)
        plt.legend(["Train Accuracy","Enumeration Test Accuracy", "Symbol Test Accuracy"])
    plt.savefig(os.path.join(args.outputdir,'accuracyHistory'))
    plt.close()
    
    np.save(os.path.join(args.outputdir,'testEpochs'),np.array(testEpochs))
    np.save(os.path.join(args.outputdir,'trainLossHistory'),np.array(trainLossHistory))
    np.save(os.path.join(args.outputdir,'trainAccuracyHistory'),np.array(trainAccuracyHistory))
    np.save(os.path.join(args.outputdir,'testAccuracyHistory'),np.array(testAccuracyHistory))
    np.save(os.path.join(args.outputdir,'testLossHistory'),np.array(testLossHistory))

    if args.enumeration_data and args.symbol_data:
        np.save(os.path.join(args.outputdir,'symbol_testAccuracyHistory'),np.array(symbol_testAccuracyHistory))
        np.save(os.path.join(args.outputdir,'symbol_testLossHistory'),np.array(symbol_testLossHistory))

    
    print(f"Test epochs: {testEpochs}")
    print(f"Train Loss: {trainLossHistory}")
    print(f"Train Accuracy: {trainAccuracyHistory}")
    print(f"Test Accuracy: {testAccuracyHistory}")
    print(f"Test Loss: {testLossHistory}")
    if args.enumeration_data and args.symbol_data:
        print(f"Symbol Test Accuracy: {symbol_testAccuracyHistory}")
        print(f"Symbol Test Loss: {symbol_testLossHistory}")

    accuracy_csv = createAccuracyCSV(args.outputdir,testEpochs)
    output_string = 'train,'+listToString(trainAccuracyHistory)
    accuracy_csv.write(output_string+'\n')
    output_string = 'test,'+listToString(testAccuracyHistory)
    accuracy_csv.write(output_string+'\n')
    accuracy_csv.flush()



if __name__ == '__main__':

    start_time = time.time()

        #Command Line Arguments 
    parser = argparse.ArgumentParser(description='Which color has more? Network Training')
    parser.add_argument('--batch-size', type=int, default=100, metavar='BS',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='TBS',
                        help='input batch size for testing (default: 200)')
    parser.add_argument('--test-interval', type=int, default=5, metavar='E',
                        help='number of epochs between tests')
    parser.add_argument('--initial-test-period', type=int, default=0, metavar='E',
                        help='number of epochs at beginning of training to test every time')
    parser.add_argument('--epochs', type=int, default=15, metavar='E',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--model', type=str, default='', metavar='M',
                        help='neural net model to use (resnet, alexnet, vgg, squeezenet, densenet,cornet_s)')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='initialize model at trained ImageNet weights')
    parser.add_argument('--feature-extract', action='store_true', default=False,
                        help='do not train the whole network just the last classification layer. Will automatically set "pretrained" to true.')        
    parser.add_argument('--replace-classifier', action='store_true', default=False,
                        help='Replace everything after the last convolutional layer with a new classifier defined using hidden-layer-size, num-hidden-layers,and num-classes')        
    parser.add_argument('--continue-training', action='store_true', default=False,
                        help='Resume training of the model pointed to by args.modeldir.')        
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='same as feature-extract but the whole network is trained.')        
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='save a copy of the model at each test step')        
    parser.add_argument('--hidden-layer-size', type=int, default=4096,
                        help='Size of the two hidden layers for the classification task.')        
    parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='NHL',
                        help='number of hidden layers in new classifier (default 1)')
    parser.add_argument('--dropout', type=float, default=0.75, metavar='dr',
                        help='decimal of neurons to dropout (default: 0.75)')
    parser.add_argument('--num-classes', type=int, default=9, metavar='Cl',
                        help='number of classes (default: 9)')
    parser.add_argument('--lowest-number', type=int, default=1, metavar='Cl',
                        help='The lowest number in the dataset (subtracted from the outputs to get the ground truth index in the output array')
    parser.add_argument('--optimizer', type=str, default='SGD', metavar='O',
                        help='optimization algorithm to use (default: SGD other options: adam)')   
    parser.add_argument('--fc-lr', type=float, default=0.005, metavar='FCLR',
                        help='learning rate for pretraining of classification layers(default: 0.005)')
    parser.add_argument('--fc-epochs', type=int, default=0, metavar='FCE',
                        help='Number of epochs to train just the classification layers (default: 0)')
    parser.add_argument('--fc-lr-decay', type=float, default=1, metavar='FCLRD',
                        help='learning rate decay for fc pretraining (default: 1)')
    parser.add_argument('--fc-lr-decay-interval', type=int, default=5, metavar='FCLRDI',
                        help='decay learning rate every x epochs (default: 5) for fc pretraining')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--lr-decay', type=float, default=1, metavar='LRD',
                        help='learning rate decay (default: 1)')
    parser.add_argument('--lr-decay-interval', type=int, default=5, metavar='LRDI',
                        help='decay learning rate every x epochs (default: 5)')
    parser.add_argument('--momentum', type=float, default=0.2, metavar='m',
                        help='SGD momentum (default: 0.2)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 2)')
    parser.add_argument('--randomize', action='store_true', default=False,
                        help='dont set random seed')
    parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-train-log', action='store_true', default=False,
                        help='supressing the training print out')
    parser.add_argument('--no-test-log', action='store_true', default=False,
                        help='supressing the testing print out')
    parser.add_argument('--indiv-accuracies', action='store_true', default=False,
                        help='Print individual accuracies for each class during testing')
    parser.add_argument('--save-outputs', action='store_true', default=False,
                        help='save the testing outputs of the net')
    parser.add_argument('--confusion-matrices', action='store_true', default=False,
                        help='calculate the confusion matrices')
    parser.add_argument('--save-layers', nargs='+', default=[], metavar='I',
                        help='layers create activation csv files for')
    parser.add_argument('--comparison-data', type=str, default='', metavar='I',
                        help='folder in data/stimuli to use for comparison data')
    parser.add_argument('--modeldir', type=str, default='', metavar='MD',
                        help='path from data/outputs folder where model is located')
    parser.add_argument('--symbol-data', type=str, default='', metavar='I',
                        help='folder in data/stimuli to use for symbol data')
    parser.add_argument('--enumeration-data', type=str, default='', metavar='I',
                        help='folder in data/stimuli to use for enumeration data')
    parser.add_argument('--estimation-data', type=str, default='', metavar='I',
                        help='folder in data/stimuli to use for estimation data')
    parser.add_argument('--solitaire-data', type=str, default='', metavar='I',
                        help='folder in data/stimuli to use for solitaire data')
    parser.add_argument('--dewind-data', type=str, default='', metavar='I',
                        help='folder in data/stimuli to use for solitaire data')
    parser.add_argument('--num-workers', type=int, default=4, metavar='W',
                        help='number of parallel batches to process. Rule of thumb 4*num_gpus (default: 4)')
    parser.add_argument('--multi-gpu', action='store_true', default=False,
                        help='run model on multiple gpus')    
#    parser.add_argument('--subset-interval', type=int, default=10, metavar='',
#                        help='Epoch intervals at which a new subset of training data is used. Training data is partitioned into epochs/subset-interval partitions of equal size')


 

    args = parser.parse_args()
    # reconcile arguments
    if args.feature_extract:
        args.pretrained == True
    print('running with args:')
    print(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('using cuda')

    main(args)

    print('Total Run Time:')
    print("--- %s seconds ---" % (time.time() - start_time))
