# Fractaldb weights are in Google Drive linked in https://github.com/hirokatsukataoka16/FractalDB-Pretrained-ResNet-PyTorch
import torch
import torch.nn as nn
import os
from zipfile import ZipFile
import math

class bn_AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(bn_AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def alexnet_fractaldb():
    model = bn_AlexNet()
    checkpoint_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fractal_weights', 'FractalDB-1000_bn_alexnet.pth')
    if not os.path.exists(checkpoint_path):
        zip_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fractal_weights.zip')
        extract_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fractal_weights')
        with ZipFile(zip_path, 'r') as fractalzip:
            fractalzip.extractall(extract_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model

