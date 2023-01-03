"""
This file is an implementation of the Depthwise Separable Convolution from:
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/abs/1610.02357
"""

import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=1, kernel_size=3, stride=1, bias=False):
        super(SeparableConv2d, self).__init__()
        in_channels = int(in_channels*alpha)
        out_channels = int(out_channels*alpha)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, bias=bias, stride=stride, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, bias=bias)
        
        self.bn_depth = nn.BatchNorm2d(num_features=in_channels)
        self.bn_point = nn.BatchNorm2d(num_features=out_channels)
        self.relu     = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn_depth(self.depthwise(x)))
        out = self.relu(self.bn_point(self.pointwise(out)))
        return out