import torch
import torch.nn as nn

class IdentityNet(nn.Module):

    def __init__(self):
        super(IdentityNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Identity()
        )
            
        
    def forward(self, x):
        return self.classifier(torch.flatten(x, start_dim=1))
