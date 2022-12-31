"""
Class provides functionality for inference using a trained model
"""
import torch

class Inference:
    def __init__(self, model=None, model_path=None, device='cpu') -> None:
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self.load_model(model_path)
        self.model.eval()
        self.device = device
    
    def infer(self, inputs):
        with torch.no_grad():
            outputs = self.model(inputs.to(self.device))
        return outputs
    
    def load_model(self, PATH):
        return torch.load(PATH)