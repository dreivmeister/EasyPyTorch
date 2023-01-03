"""
This file is an implementation of the GoogLeNet-model from:
Going Deeper with Convolutions
https://arxiv.org/abs/1409.4842
"""

import torch
import torch.nn as nn

import operator
import functools

class InceptionBlock(nn.Module):
    def __init__(self, num_input_ch, num_output_ch):
        #num_output_ch is a list of out_channels for each conv layer
        #[conv_1_1,conv_3_red,conv_3,conv_5_red,conv_5,conv_pool]
        super().__init__()
        
        self.relu = nn.ReLU()
        
        self.conv_1_1 = nn.Conv2d(in_channels=num_input_ch,out_channels=num_output_ch[0],kernel_size=1,stride=1)
        self.conv_3_red = nn.Conv2d(in_channels=num_input_ch,out_channels=num_output_ch[1],kernel_size=1,stride=1)
        self.conv_5_red = nn.Conv2d(in_channels=num_input_ch,out_channels=num_output_ch[3],kernel_size=1,stride=1)
        self.conv_pool = nn.Conv2d(in_channels=num_input_ch,out_channels=num_output_ch[5],kernel_size=1,stride=1)
        
        self.conv_3 = nn.Conv2d(in_channels=num_output_ch[1],out_channels=num_output_ch[2],kernel_size=3,stride=1, padding='same')
        self.conv_5 = nn.Conv2d(in_channels=num_output_ch[3],out_channels=num_output_ch[4],kernel_size=5,stride=1, padding='same')
        
        self.m_pool   = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
    
    def forward(self, x):
        #x - after maxpool
        out_3_red = self.relu(self.conv_3_red(x))
        out_5_red = self.relu(self.conv_5_red(x))
        out_m_p   = self.m_pool(x)
        
        out_1_1   = self.relu(self.conv_1_1(x))
        out_3     = self.relu(self.conv_3(out_3_red))
        out_5     = self.relu(self.conv_5(out_5_red))
        out_pool  = self.relu(self.conv_pool(out_m_p))
        
        #depth concat and return
        return torch.cat((out_1_1,out_3,out_5,out_pool),dim=1)

#from Inceptionv2 paper
class InceptionBlock1(nn.Module):
    def __init__(self, num_input_ch, num_output_ch):
        #num_output_ch is a list of out_channels for each conv layer
        #[conv_1_1,conv_3_red,conv_3,conv_5_red,conv_5,conv_pool]
        super().__init__()
        
        self.relu = nn.ReLU()
        
        self.conv_1_1 = nn.Conv2d(in_channels=num_input_ch,out_channels=num_output_ch[0],kernel_size=1,stride=1)
        self.conv_3_red = nn.Conv2d(in_channels=num_input_ch,out_channels=num_output_ch[1],kernel_size=1,stride=1)
        self.conv_3_3_red = nn.Conv2d(in_channels=num_input_ch,out_channels=num_output_ch[3],kernel_size=1,stride=1)
        self.conv_pool = nn.Conv2d(in_channels=num_input_ch,out_channels=num_output_ch[5],kernel_size=1,stride=1)
        
        self.conv_3 = nn.Conv2d(in_channels=num_output_ch[1],out_channels=num_output_ch[2],kernel_size=3,stride=1, padding='same')
        self.conv_3_3_1 = nn.Conv2d(in_channels=num_output_ch[3],out_channels=num_output_ch[4],kernel_size=3,stride=1, padding='same')
        self.conv_3_3_2 = nn.Conv2d(in_channels=num_output_ch[4],out_channels=num_output_ch[4],kernel_size=3,stride=1, padding='same')

        self.m_pool   = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
    
    def forward(self, x):
        #x - after maxpool
        out_3_red   = self.relu(self.conv_3_red(x))
        out_3_3_red = self.relu(self.conv_3_3_red(x))
        out_m_p     = self.m_pool(x)
        
        out_1_1   = self.relu(self.conv_1_1(x))
        out_3     = self.relu(self.conv_3(out_3_red))
        out_3_3   = self.relu(self.conv_3_3_2(self.relu(self.conv_3_3_1(out_3_3_red))))
        out_pool  = self.relu(self.conv_pool(out_m_p))
        
        #depth concat and return
        return torch.cat((out_1_1,out_3,out_3_3,out_pool),dim=1)


class AuxiliaryClassifier(nn.Module):
    def __init__(self, num_classes, input_dim):
        super().__init__()
        
        self.relu    = nn.ReLU()
        
        self.conv_l = nn.Sequential(
            nn.AvgPool2d(kernel_size=5,stride=3),
            nn.Conv2d(in_channels=input_dim[0],out_channels=128,kernel_size=1),
        )
    
        num_dense_inp = functools.reduce(operator.mul, list(self.conv_l(torch.rand(1, *input_dim)).shape))
        
        self.lin_1   = nn.Linear(in_features=num_dense_inp,out_features=1024)
        self.dropout = nn.Dropout(p=0.7)
        self.classifier = nn.Linear(in_features=1024,out_features=num_classes)
        self.softm   = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.conv_l(x)
        out = self.relu(out)
        
        bs = out.size(0)
        out = out.view(bs,-1)
        
        out = self.lin_1(out)
        out = self.relu(out)
        out = self.dropout(out)
        return self.softm(self.classifier(out))


#num_dense_inp = functools.reduce(operator.mul, list(self.conv_layers(torch.rand(1, *input_dim)).shape))
#GoogLeNet
class GoogLeNet(nn.Module):
    def __init__(self, num_classes, input_dim):
        super().__init__()
        
        
        #used multiple times
        self.m_pool = nn.MaxPool2d(kernel_size=3,stride=2)
        self.relu   = nn.ReLU()
        self.lrn    = nn.LocalResponseNorm(size=2)
        
        
        self.conv_7 = nn.Conv2d(in_channels=input_dim[0],out_channels=64,kernel_size=7,stride=2)
        self.conv_3 = nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,stride=1)
        
        #maxpool
        
        #inception
        self.inception_3 = nn.ModuleList([
            InceptionBlock(192, [64,96,128,16,32,32]),
            InceptionBlock(256, [128,128,192,32,96,64])
            ])

        #maxpool
        
        #inception
        self.inception_4 = nn.ModuleList([
            InceptionBlock(480, [192,96,208,16,48,64]),
            InceptionBlock(512, [160,112,224,24,64,64]),
            InceptionBlock(512, [128,128,256,24,64,64]),
            InceptionBlock(512, [112,144,288,32,64,64]),
            InceptionBlock(528, [256,160,320,32,128,128])
            ])
        
        #maxpool
        
        #inception
        self.inception_5 = nn.ModuleList([
            InceptionBlock(832, [256,160,320,32,128,128]),
            InceptionBlock(832, [384,192,384,48,128,128])
            ])
        
        
        #avgpool
        self.avg_pool = nn.AvgPool2d(kernel_size=2,stride=1)
        #dropout
        self.dropout  = nn.Dropout(p=0.4)
        #linear
        self.fc       = nn.Linear(in_features=1024,out_features=num_classes)
        #softmax output
        self.softm    = nn.Softmax(dim=1)
        
        self.aux_1    = AuxiliaryClassifier(num_classes, (512,12,12))
        self.aux_2    = AuxiliaryClassifier(num_classes, (528,12,12))
        self.outs = []
        
    def forward(self, x):
        out   = self.lrn(self.relu(self.conv_7(x)))
        out   = self.m_pool(out)
        
        out   = self.lrn(self.relu(self.conv_3(out)))
        out   = self.m_pool(out)
        
        #first inception layers
        for i,bl in enumerate(self.inception_3):
            out = bl(out)
        out = self.m_pool(out)
        
        #second inception layers
        for i,bl in enumerate(self.inception_4):
            if i == 1:
                self.outs.append(self.aux_1(out))
            if i == 4:
                self.outs.append(self.aux_2(out))
            out = bl(out)
        out = self.m_pool(out)
        
        #third
        for i,bl in enumerate(self.inception_5):
            out = bl(out)
        out = self.m_pool(out)
        
        #classifier
        out = self.avg_pool(out)
        out = self.dropout(out)
        bs = out.size(0)
        out = out.view(bs, -1)
        out = self.fc(out)
        
        return self.softm(out)