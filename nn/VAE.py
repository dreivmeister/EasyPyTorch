"""
This file is an implementation of the Variational Autoencoder from:
Auto-Encoding Variational Bayes
https://arxiv.org/abs/1312.6114
"""

import torch
import torch.nn.functional as F
from torch import nn, optim



class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, z_dim=20):
        super().__init__()
        
        #encoder
        #input
        self.img_2hid = nn.Linear(input_dim, hidden_dim)
        #latent space
        #mean
        self.hid_2mu = nn.Linear(hidden_dim, z_dim)
        #covariance
        self.hid_2sigma = nn.Linear(hidden_dim, z_dim)
        
        #decoder
        self.z_2hid = nn.Linear(z_dim, hidden_dim)
        #output
        self.hid_2img = nn.Linear(hidden_dim, input_dim)
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        #q_phi(z|x)
        #x - input
        hidden_repr = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(hidden_repr), self.hid_2sigma(hidden_repr)
        return mu, sigma
    
    def decode(self, z):
        #p_theta(x|z)
        #z - latent repr
        hidden_repr = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(hidden_repr))
    
    def criterion(self, x, x_rec, mu, sigma):
        rec_loss = nn.BCELoss(reduction='sum')(x_rec, x) # reconstruction loss
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2))-mu.pow(2)-sigma.pow(2)) # kl divergence
        return rec_loss + kl_div
        
    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_new = mu + sigma*epsilon
        x_rec = self.decode(z_new)
        return x_rec, mu, sigma
    

# def inference(digit, num_ex=5):
#     imgs = []
#     idx = 0
#     #get example of each digit
#     for x, y in dataset:
#         if y == idx:
#             imgs.append(x)
#             idx += 1
#         if idx == 10:
#             break
    
#     #get mu and sigma for each digit
#     encodings_digit = []
#     for d in range(10):
#         with torch.no_grad():
#             mu, sigma = model.encode(imgs[d].view(1,784))
#         encodings_digit.append((mu,sigma))
    
#     mu, sigma = encodings_digit[digit]
#     for ex in range(num_ex):
#         epsilon = torch.randn_like(sigma)
#         z = mu + sigma*epsilon
#         out = model.decode(z)
#         out = out.view(-1, 1, 28, 28)
#         save_image(out, f"{digit}_{ex}.jpg")