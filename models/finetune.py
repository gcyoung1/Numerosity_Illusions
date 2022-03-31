import torch
torch.set_printoptions(threshold=5000)
import torch.nn as nn
from torchvision import models
import sys
sys.path.append('../CORnet/')
import cornet
#from cornet.cornet_s import CORnet_S

class Finetune(nn.Module):
    def __init__(self,model_name,num_hidden_layers,hidden_layer_size,dropout,num_classes):
        super(Finetune,self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.num_classes = num_classes
        self.model_name = model_name
        self.dropout=dropout


        if self.model_name == "resnet":
            model = models.resnet18(pretrained=True)
            self.final_layer_size = model.fc.in_features*7*7
            self.features = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4
            )
        elif self.model_name == "alexnet":
            model = models.alexnet(pretrained=True)
            self.final_layer_size = 256*6*6
            self.avgpool = model.avgpool
            self.features = model.features        
        elif self.model_name == "vgg":
            model = models.vgg11_bn(pretrained=True)
            self.final_layer_size = 512*7*7
            self.features = model.features        
        elif self.model_name == "densenet":
            model = models.densenet121(pretrained=True)
            self.final_layer_size = model.classifier.in_features
            self.features = model.features        
        elif self.model_name == "cornet_s":
            model = cornet.cornet_s(pretrained=True)
            # model = CORnet_S()#
            # hash = '1d3f7974'
            # url = f'https://s3.amazonaws.com/cornet-models/cornet_s-{hash}.pth'
            # ckpt_data = torch.utils.model_zoo.load_url(url, map_location=None)
            # model.load_state_dict(ckpt_data['state_dict'])

            self.final_layer_size = 7*7*512
            self.features = nn.Sequential(
                model.module.V1,
                model.module.V2,
                model.module.V4,
                model.module.IT
            )

        else:
            print("Invalid model name, exiting...")
            exit()


        if self.num_hidden_layers == 1:

            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.final_layer_size,self.hidden_layer_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_layer_size, self.num_classes)
            )

        else:

            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.final_layer_size,self.hidden_layer_size),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_layer_size,self.hidden_layer_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_layer_size, self.num_classes)
            )
        
    def forward(self, x):
        x = self.features(x)
        if self.model_name == "alexnet":
            x = self.avgpool(x)
        x=torch.flatten(x,1)
        output = self.classifier(x)
        return output
