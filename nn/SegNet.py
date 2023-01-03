"""
This file is an implementation of the Depthwise Separable Convolution from:
SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
https://arxiv.org/abs/1511.00561
"""

import torch.nn as nn


class SegNetEncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,padding=padding)
        self.bn   = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
    
class SegNetDecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding)
        self.bn   = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SegNet(nn.Module):
    def __init__(self, num_classes, input_dim):
        super().__init__()
        #Encoder
        self.encoder_1 = nn.Sequential(
            SegNetEncBlock(in_channels=3,out_channels=64),
            SegNetEncBlock(in_channels=64,out_channels=64),
        )
        self.mpool_1   = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
        
        self.encoder_2 = nn.Sequential(
            SegNetEncBlock(in_channels=64,out_channels=128),
            SegNetEncBlock(in_channels=128,out_channels=128),
        )
        self.mpool_2   = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
        
        self.encoder_3 = nn.Sequential(
            SegNetEncBlock(in_channels=128,out_channels=256),
            SegNetEncBlock(in_channels=256,out_channels=256),
            SegNetEncBlock(in_channels=256,out_channels=256),
        )
        self.mpool_3   = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
        
        self.encoder_4 = nn.Sequential(
            SegNetEncBlock(in_channels=256,out_channels=512),
            SegNetEncBlock(in_channels=512,out_channels=512),
            SegNetEncBlock(in_channels=512,out_channels=512),
        )
        self.mpool_4   = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
        
        self.encoder_5 = nn.Sequential(
            SegNetEncBlock(in_channels=512,out_channels=512),
            SegNetEncBlock(in_channels=512,out_channels=512),
            SegNetEncBlock(in_channels=512,out_channels=512),
        )
        self.mpool_5   = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
        
        
        
        #Decoder
        self.unmpool_5 = nn.MaxUnpool2d(kernel_size=2,stride=2)
        self.decoder_5 = nn.Sequential(
            SegNetDecBlock(in_channels=512,out_channels=512),
            SegNetDecBlock(in_channels=512,out_channels=512),
            SegNetDecBlock(in_channels=512,out_channels=512),
        )
        
        self.unmpool_4 = nn.MaxUnpool2d(kernel_size=2,stride=2)
        self.decoder_4 = nn.Sequential(
            SegNetDecBlock(in_channels=512,out_channels=512),
            SegNetDecBlock(in_channels=512,out_channels=512),
            SegNetDecBlock(in_channels=512,out_channels=256),
        )
        
        self.unmpool_3 = nn.MaxUnpool2d(kernel_size=2,stride=2)
        self.decoder_3 = nn.Sequential(
            SegNetDecBlock(in_channels=256,out_channels=256),
            SegNetDecBlock(in_channels=256,out_channels=256),
            SegNetDecBlock(in_channels=256,out_channels=128),
        )
        
        self.unmpool_2 = nn.MaxUnpool2d(kernel_size=2,stride=2)
        self.decoder_2 = nn.Sequential(
            SegNetDecBlock(in_channels=128,out_channels=128),
            SegNetDecBlock(in_channels=128,out_channels=64),
        )
        
        self.unmpool_1 = nn.MaxUnpool2d(kernel_size=2,stride=2)
        self.decoder_1 = nn.Sequential(
            SegNetDecBlock(in_channels=64,out_channels=64),
            SegNetDecBlock(in_channels=64,out_channels=num_classes),
        )
        
        self.softm = nn.Softmax2d()
    
    def forward(self, x):
        
        #Encoder
        out_enc_1 = self.encoder_1(x)
        out_mp_1,ind_mp_1  = self.mpool_1(out_enc_1)
        
        out_enc_2 = self.encoder_2(out_enc_1)
        out_mp_2,ind_mp_2  = self.mpool_2(out_enc_2)
        
        out_enc_3 = self.encoder_3(out_enc_2)
        out_mp_3,ind_mp_3  = self.mpool_3(out_enc_3)
        
        out_enc_4 = self.encoder_4(out_enc_3)
        out_mp_4,ind_mp_4  = self.mpool_4(out_enc_4)
        
        out_enc_5 = self.encoder_5(out_enc_4)
        out_mp_5,ind_mp_5  = self.mpool_5(out_enc_5) 
        
        #Decoder
        out_unmp_5 = self.unmpool_5(out_mp_5,ind_mp_5)
        out_dec_5  = self.decoder_5(out_unmp_5)
        
        out_unmp_4 = self.unmpool_4(out_mp_4,ind_mp_4)
        out_dec_4  = self.decoder_4(out_unmp_4)
        
        out_unmp_3 = self.unmpool_3(out_mp_3,ind_mp_3)
        out_dec_3  = self.decoder_3(out_unmp_3)
        
        out_unmp_2 = self.unmpool_2(out_mp_2,ind_mp_2)
        out_dec_2  = self.decoder_2(out_unmp_2)
        
        out_unmp_1 = self.unmpool_1(out_mp_1,ind_mp_1)
        out_dec_1  = self.decoder_1(out_unmp_1)
        
        return self.softm(out_dec_1)
    
