import torch
import torch.nn as nn
import torchvision
from src.data import Data
from src.training import Training

# inputs = torch.randn(10, 3, 28, 28)
# labels = torch.randint(0, 10, (10,1))

if __name__ == '__main__':

    net = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3),
        nn.BatchNorm2d(num_features=32),
        nn.ReLU(),
        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
        nn.BatchNorm2d(num_features=64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3),
        nn.BatchNorm2d(num_features=128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.Flatten(),
        nn.Linear(in_features=21632,out_features=10),
        nn.Softmax(dim=1)
    )

    optimizer = torch.optim.Adam(params=net.parameters())
    criterion = nn.CrossEntropyLoss()


    D = Data(batch_size=12, dataset=torchvision.datasets.CIFAR10(root='\datasets',download=True,transform=torchvision.transforms.ToTensor()))
    data_loader = D.dataloader


    T = Training(net, optimizer, criterion)

    T.fit(num_epochs=1,train_loader=data_loader,plot=True)