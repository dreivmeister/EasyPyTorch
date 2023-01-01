"""
Class which provides functionality to train and test a PyTorch Network.
Easy to setup and modify, verbose training progress vis.
"""
#TODO: write generic fit function for variable number of optimizers etc. and arbitrary attachments

# Imports
import torch

class Training:
    def __init__(self, model, optimizer, criterion, device='cpu', attachments=None) -> None:
        self.model = model # PyTorch model object
        self.optimizer = optimizer #hyperparams are give through it for now
        self.criterion = criterion
        self.device = device # add GPU func
        self.model = self.model.to(self.device)
        
        self.attachments = attachments # list of functions to run during training
        
    
    def fit(self, num_epochs, train_loader, log_inter=10, plot=False):
        losses = []
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):                
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                if i % log_inter == 0:
                    norm_run_loss = running_loss / log_inter
                    losses.append(norm_run_loss)
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {norm_run_loss:.3f}')
                    running_loss = 0.0
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(range(len(losses)), losses, '-ok');
            plt.show()
    
    
    def predict(self, test_loader):
        running_test_loss = 0.0
        num_samples = 0
        with torch.no_grad():
            for _, data in enumerate(test_loader,0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                
                outputs = self.model(inputs)
                running_test_loss += self.criterion(outputs, labels)
                num_samples += labels.size(0)
        resulting_test_loss = running_test_loss / num_samples
        print(f' test loss: {resulting_test_loss:.3f}')


    def save_model(self, PATH):
        torch.save(self.model, PATH)
                
                
                
    