"""
This file is an implementation of the VGG11-Model from:
Very Deep Convolutional Networks for Large-Scale Image Recognition
https://arxiv.org/abs/1409.1556
"""


import torch
import torch.nn as nn

import operator
import functools


class VGG11(nn.Module):
    def __init__(self, num_classes, input_dim):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_dim[0],out_channels=64,kernel_size=3,padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
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
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        batch_size = x.size(0)

        out = self.conv_layers(x)
        out = out.view(batch_size, -1)  # flatten the vector
        out = self.dense_layers(out)
        return out