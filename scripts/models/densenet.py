import torch
import torch.nn as nn

class DenseNet(nn.Module):

    def __init__(self):
        super(DenseNet, self).__init__()
        latent_size = 4096
        self.classifier = nn.Sequential(
            nn.Linear(224*224*3, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
        )
            
        
    def forward(self, x):
        return self.classifier(torch.flatten(x, start_dim=1))
