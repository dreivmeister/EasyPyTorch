

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#Network Architecture Config
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


#Network Architecture
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs)
        self.batch_norm_layer = nn.BatchNorm2d(out_channels)
        self.activation_function = nn.LeakyReLU(0.1)
        self.use_batch_norm = use_batch_norm
        
    def forward(self, x):
        if self.use_batch_norm:
            return self.activation_function(self.batch_norm_layer(self.conv_layer(x)))
        else:
            return self.conv_layer(x)
    
    
class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.use_residual = use_residual
        self.num_repeats = num_repeats
        
        for _ in range(self.num_repeats):
            self.layers += [nn.Sequential(CNNBlock(channels, channels // 2, kernel_size=1),
                                          CNNBlock(channels // 2, channels, kernel_size=3, padding=1))]
            
    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return x
    
    
class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, use_batch_norm=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )
    
    
class YOLOV3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = self._create_conv_layers()
        
    def forward(self, x):
        multiscale_outputs = []
        route_connections = []
        
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                multiscale_outputs.append(layer(x))
                continue
                
            #standard
            x = layer(x)
            
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
                
        return multiscale_outputs
                
    
    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        
        for module in config:
            
            #standard cnn block
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))
                
            
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3
        return layers