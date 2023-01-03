"""
This file is an implementation of the DeepLabV3+-Model from:
Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
https://arxiv.org/abs/1802.02611
"""

import torch
import torch.nn as nn
import torchvision

class ASPP(nn.Module): # atrous separable convolution
    def __init__(self, inp_chs):
        super().__init__()
        
        #input_dim - (channels, height, width)
        #defining the operations
        self.img_pool     = nn.AvgPool2d(kernel_size=2)
        self.upsample     = nn.Upsample(scale_factor=2)
        self.conv_1_1     = nn.Conv2d(in_channels=inp_chs,out_channels=256,kernel_size=1)
        self.conv_dil_6   = nn.Conv2d(in_channels=inp_chs,out_channels=256,kernel_size=3,dilation=6,padding='same')
        self.conv_dil_12  = nn.Conv2d(in_channels=inp_chs,out_channels=256,kernel_size=3,dilation=12,padding='same')
        self.conv_dil_18  = nn.Conv2d(in_channels=inp_chs,out_channels=256,kernel_size=3,dilation=18,padding='same')
        
    def forward(self, x):
        img_pool_out = self.img_pool(x)
        img_pool_out = self.upsample(img_pool_out)
        conv_1_1 = self.conv_1_1(x)
        conv_dil_6 = self.conv_dil_6(x)
        conv_dil_12 = self.conv_dil_12(x)
        conv_dil_18 = self.conv_dil_18(x)
        
        return torch.cat((img_pool_out,conv_1_1,conv_dil_6,conv_dil_12,conv_dil_18),dim=1)
    
    
class Decoder(nn.Module):
    def __init__(self,inp_chs_interm, inp_chs_end,num_classes):
        super().__init__()
        #interm out 1x1 conv
        self.conv_1_1_interm = nn.Conv2d(in_channels=inp_chs_interm,out_channels=256,kernel_size=1)
        #end output upsample
        self.upsample = nn.Upsample(scale_factor=2)
        #output concat
        #3x3 conv and upsample to input size
        self.conv_3_3 = nn.Conv2d(in_channels=256+inp_chs_end,out_channels=256,kernel_size=3,padding='same')
        
        #get output preds
        self.conv_1_1_pred = nn.Conv2d(in_channels=256,out_channels=num_classes,kernel_size=1)
        self.upsample_2    = nn.Upsample(scale_factor=8)
        
    def forward(self, interm_out, end_out):
        interm_out = self.conv_1_1_interm(interm_out)
        end_out    = self.upsample(end_out)
        interm_end_concat = torch.cat((interm_out,end_out),dim=1)
        interm_end_concat = self.conv_3_3(interm_end_concat)
        return self.conv_1_1_pred(self.upsample_2(interm_end_concat))
    
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.interm_layer = 'layer3'
        self.activation = {}
        
        #load model
        self.resnet50 = torchvision.models.resnet50(pretrained=True,replace_stride_with_dilation=[False,True,False])
        #remove top (fc and classifier)
        self.resnet50 = torch.nn.Sequential(*(list(self.resnet50.children())[:-2]))
        #register hook for intermediate activation
        self.resnet50[6].register_forward_hook(self.get_activation(self.interm_layer))
        
        #
        self.aspp = ASPP(inp_chs=2048)
        self.conv_1_1 = nn.Conv2d(in_channels=3072,out_channels=48,kernel_size=1)
    
    def get_activation(self,name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def forward(self, x):
        #output of last block which aspp'd
        end_output = self.resnet50(x)
        end_output = self.aspp(end_output)
        end_output = self.conv_1_1(end_output)
        
        #output of intermediate block
        interm_output = self.activation[self.interm_layer]
    
        return interm_output, end_output


class DeepLabV3p(nn.Module):
    def __init__(self, num_classes, inp_chs_interm=1024, inp_chs_end=48):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(inp_chs_interm, inp_chs_end, num_classes)
    def forward(self, x):
        enc_out = self.encoder(x)
        return self.decoder(enc_out[0],enc_out[1])