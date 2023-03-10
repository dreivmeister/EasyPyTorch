import unittest
import torch
import torch.nn as nn
from src.inference import Inference


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



class InferenceClassTest(unittest.TestCase):
    def create_class(self):
        I = Inference(model=net)


# if __name__=='__main__':
#     unittest.main()