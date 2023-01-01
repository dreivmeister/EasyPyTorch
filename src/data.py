"""
Class which provides functionality to load image data (for now) into datasets and datalaoders.
"""
import torch
import torch.nn as nn
import torchvision



class Data:
    def __init__(self, batch_size, dataset=None, dataset_path=None) -> None:
        #dataset_name - name of a torchvision dataset
        #dataset_path - path of a local image dataset with specified structure
        if dataset is not None:
            self.dataset = dataset
            self.dataloader = self.get_dataloader_from_dataset(self.dataset, batch_size=batch_size)
        elif dataset_path is not None:
            self.dataset_path = dataset_path
            self.dataset = self.get_custom_dataset(self.dataset_path)
            self.dataloader = self.get_dataloader_from_dataset(self.dataset, batch_size=batch_size)
        
    
    def get_dataloader_from_dataset(self, dataset, batch_size, shuffle=True, num_workers=2):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    def get_custom_dataset(self, root):
        """
        structure:
        
        directory/
        ├── class_x
        │   ├── xxx.ext
        │   ├── xxy.ext
        │   └── ...
        │       └── xxz.ext
        └── class_y
            ├── 123.ext
            ├── nsdf3.ext
            └── ...
            └── asd932_.ext
        """
        dataset = torchvision.datasets.DatasetFolder(root=root)
    
    def get_data_batch(self):
        #returns a random batch of data
        batch = iter(self.dataloader)
        return next(batch)