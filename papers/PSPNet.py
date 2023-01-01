"""
This file is an implementation of the PSPNet-Model from:
Pyramid Scene Parsing Network
https://arxiv.org/abs/1612.01105
"""


import torch
import torch.nn as nn
from ResNet import ResNet18


class PyramidPoolingModule(nn.Module):
    def __init__(self, bin_sizes, feature_map_shape):
        super().__init__()
        #bin_sizes - list of kernel sizes for pooling (integers-assuming square kernels)
        #scale_factor - the factor by which the pooled maps are upsampled
        #init the pooling ops and the upsampling
        self.bin_sizes = bin_sizes
        self.av_pools = nn.ModuleList([nn.AvgPool2d(kernel_size=s) for s in self.bin_sizes])
        self.conv_lays = nn.ModuleList([nn.Conv2d(in_channels=feature_map_shape[0],out_channels=1,kernel_size=1) for _ in range(len(self.bin_sizes))])
        self.upsample_lay = nn.Upsample(size=feature_map_shape[1:])
        
    
    def forward(self, x):
        #x - is the feature map
        context_maps = []
        
        for i in range(len(self.bin_sizes)):
            context_maps.append(self.upsample_lay(self.conv_lays[i](self.av_pools[i](x))))
        
        return context_maps


class PSPNet(nn.Module):
    def __init__(self, num_classes, input_dim, bin_sizes=[1,2,3,6]):
        super().__init__()
        
        #two parts:
        #1. CNN (ResNet type for feature extraction, provides Feature Map)
        self.CNN = ResNet18(image_channels=input_dim[0])
        with torch.no_grad():
            x = torch.randn(1,input_dim[0],input_dim[1],input_dim[2])
            out_s = self.CNN(x).shape[1:]
        #2. PPM (Average Pooling at multiple sizes, upsampling and concatenation)
        #assuming bin_size divides feature_map size
        self.PPM = PyramidPoolingModule(bin_sizes=bin_sizes, feature_map_shape=out_s)
    
        #3. Final Prediction Layer
        self.prediction_lay = nn.Conv2d(in_channels=out_s[0]+len(bin_sizes),out_channels=num_classes,kernel_size=1)
        
    
    
    def forward(self, x):
        _,feature_map = self.CNN(x)
        context_maps = self.PPM(feature_map)
        concat_map = torch.cat((feature_map, *context_maps),dim=1)
        return self.prediction_lay(concat_map)
