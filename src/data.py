"""
Class which provides functionality to load image data (for now) into datasets and datalaoders.
"""
import torch
import torch.nn as nn
import torchvision



class Data:
    def __init__(self, dataset_name=None, dataset_path=None) -> None:
        #dataset_name - name of a torchvision dataset
        #dataset_path - path of a local image dataset with specified structure
        pass
    
    def load_dataset_from_name(self):
        pass
    
    def load_dataset_from_path(self):
        pass
    
    