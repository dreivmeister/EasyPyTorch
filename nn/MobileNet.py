"""
This file is an implementation of the Depthwise Separable Convolution from:
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
https://arxiv.org/abs/1704.04861
"""

import torch.nn as nn
from DSC import SeparableConv2d

class MobileNet(nn.Module):
    def __init__(self, num_classes, alpha=1):
        super().__init__()
        
        
        #default for SC: ks=3,s=1
        #MobileNet Architecture
        
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2),
            SeparableConv2d(in_channels=32,out_channels=64),
            SeparableConv2d(in_channels=64,out_channels=128,stride=2),
            SeparableConv2d(in_channels=128,out_channels=128),
            SeparableConv2d(in_channels=128,out_channels=256,stride=2),
            SeparableConv2d(in_channels=256,out_channels=256),
            SeparableConv2d(in_channels=256,out_channels=512,stride=2),
        )

        
        self.SC_block = nn.Sequential(
            SeparableConv2d(in_channels=512,out_channels=512),
            SeparableConv2d(in_channels=512,out_channels=512),
            SeparableConv2d(in_channels=512,out_channels=512),
            SeparableConv2d(in_channels=512,out_channels=512),
            SeparableConv2d(in_channels=512,out_channels=512),
        )
        
        self.SC_7  = SeparableConv2d(in_channels=512,out_channels=1024,stride=2)
        self.SC_8  = SeparableConv2d(in_channels=1024,out_channels=1024,stride=2)
        #weird that kernel size doesnt fit
        self.av_pool = nn.AvgPool2d(kernel_size=4)
        self.FC    = nn.Linear(in_features=1024,out_features=num_classes)
        self.softm = nn.Softmax(dim=1)
    
    def forward(self, x):
        out_inp = self.input_block(x)
        out_block = self.SC_block(out_inp)
        out_conv = self.SC_8(self.SC_7(out_block))
        out_pool = self.av_pool(out_conv)
        out_pool = out_pool.view(out_pool.size(0),-1) # flatten
        return self.softm(self.FC(out_pool))