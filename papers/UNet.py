"""
This file is an implementation of the UNet-Model from:
U-Net: Convolutional Networks for Biomedical Image Segmentation
https://arxiv.org/abs/1505.04597
"""
#DONE

import torch
import torch.nn as nn
import torchvision



class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3)
        self.relu  = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))
    

class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i],chs[i+1]) for i in range(len(chs)-1)])
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs
    

class Decoder(nn.Module):
    def __init__(self, chs=(1024,512,256,128,64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i],chs[i+1],2,2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i],chs[i+1]) for i in range(len(chs)-1)])
    
    def forward(self, x, ftrs):
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            
            enc_ftr = self.crop(ftrs[i], x)
            x = torch.cat([x,enc_ftr],dim=1) #concat on channel dim
            x = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftr, x):
        _,_,H,W = x.shape
        enc_ftr = torchvision.transforms.CenterCrop([H,W])(enc_ftr)
        return enc_ftr
    

class UNet(nn.Module):
    def __init__(self, inp_chs=3, enc_chs=(64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=2, retain_dim=False, out_sz=(572,572)):
        super().__init__()
        enc_chs = (inp_chs,) + enc_chs #add input channels to channel progression of encoder
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, kernel_size=1) #output map (1x1 convolution)
        self.retain_dim  = retain_dim
        self.out_sz      = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:]) #first of reverse is last one added
        out      = self.head(out)
        if self.retain_dim:
            out = torch.functional.interpolate(out, self.out_sz)
        return out