import torch
import torch.nn as nn
from torchvision.models import resnet18
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x
    

class PrimNet(nn.Module):
    def __init__(self,embedding_size,num_classes,pretrained=False):
        super(PrimNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, groups = 32),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, groups = 32),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, groups = 32),
            nn.BatchNorm2d(1024),
            nn.ReLU())
        self.fc1 = nn.Linear(7*7*1024, embedding_size)
        self.fc2 = nn.Linear(embedding_size, num_classes)
    
   
    
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output
    
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = channel_shuffle(out, 32)
        out = self.layer3(out)
        out = channel_shuffle(out, 32)
        out = self.layer4(out)
        out = channel_shuffle(out, 32)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        self.features = self.l2_norm(out)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha=10
        self.features = self.features*alpha
        return self.features
    
    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.fc2(features)
        return res
    
   
    
 

   