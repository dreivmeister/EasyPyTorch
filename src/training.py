"""
Class which provides functionality to train and test a PyTorch Network.
Easy to setup and modify, verbose training progress vis.
"""
# Imports
import torch
import torch.nn as nn



class Training:
    def __init__(self, model, optimizer, criterion, device) -> None:
        self.model = model # PyTorch model object
        self.optimizer = optimizer #hyperparams are give through it for now
        self.criterion = criterion
        self.device = device # add GPU func
        
        
    def fit(self, num_epochs, train_loader, log_inter):
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                if i % log_inter == 0:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
    
    
    def predict(self, test_loader):
        running_test_loss = 0.0
        num_samples = 0
        with torch.no_grad():
            for _, data in enumerate(test_loader,0):
                inputs, labels = data
                
                outputs = self.model(inputs)
                running_test_loss += self.criterion(outputs, labels)
                num_samples += labels.size(0)
        resulting_test_loss = running_test_loss / num_samples
        print(f' test loss: {resulting_test_loss:.3f}')
    
    
    def evaluate(self, inputs):
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs
    
    
        
    
                
                
                
                
                
    