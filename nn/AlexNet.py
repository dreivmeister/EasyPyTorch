"""
This file is an implementation of the AlexNet-Model from:
ImageNet Classification with Deep Convolutional Neural Networks
https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import operator
import functools


class AlexNet(nn.Module):
    #input_dim: (channels, height, width)
    def __init__(self, num_classes, input_dim):
        super().__init__() 
        
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_dim[0],out_channels=96,kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        
        num_dense_inp = functools.reduce(operator.mul, list(self.conv_layers(torch.rand(1, *input_dim)).shape))

        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=num_dense_inp,out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(inplace=True),
            nn.Linear(in_features=4096,out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(inplace=True),
            nn.Linear(in_features=4096,out_features=num_classes),
        )
        
    def forward(self, x):
        batch_size = x.size(0)

        out = self.conv_layers(x)
        out = out.view(batch_size, -1)  # flatten the vector
        out = self.dense_layers(out)
        return out