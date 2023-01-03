"""
This file is an implementation of the FCN16/32-Model from:
Fully Convolutional Networks for Semantic Segmentation
https://arxiv.org/abs/1411.4038
"""

import torch.nn as nn

class FCN_16s(nn.Module):
    def __init__(self, num_classes, input_dim):
        super().__init__()
        
        
        #vgg16 archictecture
        self.mpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU()
        )
        
        #upsampling (32-s)
        #self.location_pred = nn.Conv2d(in_channels=512,out_channels=num_classes,kernel_size=1)
        #self.upsample      = nn.Upsample(scale_factor=32,mode='bilinear')
        
        #self.conv_transpose_32 = nn.ConvTranspose2d(in_channels=512, out_channels=num_classes, kernel_size=(32,32), stride=(32,32))
        
        #upsampling (16-s)
        self.upsample_2 = nn.Upsample(scale_factor=2,mode='bilinear')
        self.conv_transpose_16 = nn.ConvTranspose2d(in_channels=512, out_channels=num_classes, kernel_size=(16,16), stride=(16,16))
        
        
        
    def forward(self, x):
        out = self.conv_1(x)
        out_1 = self.mpool(out)
        
        out = self.conv_2(out_1)
        out_2 = self.mpool(out)
        
        out = self.conv_3(out_2)
        out_3 = self.mpool(out)
        
        out = self.conv_4(out_3)
        out_4 = self.mpool(out)
        
        out = self.conv_5(out_4)
        out_5 = self.mpool(out)
  
        #upsampling (32-s)
        #out = self.conv_transpose_32(out)
        #i feel like both methods could work
        #out = self.location_pred(out)
        #out = self.upsample(out)
        
        #upsampling (16-s)
        out_5_upsampled = self.upsample_2(out_5)
        out = out_5_upsampled + out_4
        out = self.conv_transpose_16(out)
        
        
        return out