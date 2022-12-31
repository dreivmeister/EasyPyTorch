import unittest
import torch
import torch.nn as nn
from src.training import Training


inputs = torch.randn(10, 3, 28, 28)
labels = torch.randint(0, 10, (10,1))

net = nn.Sequential(
    nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(in_features=26*26*32,out_features=10),
    nn.Softmax(dim=1)
)

optimizer = torch.optim.Adam(params=net.parameters())
criterion = nn.CrossEntropyLoss()



class TrainingClassTest(unittest.TestCase):
    def create_class(self):
        T = Training(net, optimizer, criterion)
        #assert T is not None
    
    def train_model(self):
        T = Training(net, optimizer, criterion)
        T.fit(num_epochs=1)


if __name__=='__main__':
    unittest.main()
    
    
#how to run from:
#C:\Users\DELL User\Desktop\EasyPyTorch\EasyPyTorch>:
#py -m unittest tests.test_training.TrainingClassTest.create_class