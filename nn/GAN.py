"""
This file is an implementation of the CycleGAN-Model from:
Generative Adversarial Networks
https://arxiv.org/abs/1406.2661
"""


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt


"""
def get_minibatch(trainX, batch_size):
    indices = torch.randperm(trainX.shape[0])[:batch_size]
    return torch.tensor(trainX[indices], dtype=torch.float).reshape(batch_size, -1)

def sample_noise(size, dim=100):
    out = torch.empty(size, dim)
    mean = torch.zeros(size, dim)
    std = torch.ones(dim)
    torch.normal(mean, std, out=out)
    return out
"""


class Generator(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=1200, output_dim=28*28):
        super(Generator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), nn.Tanh())
    
    def forward(self, noise):
        return self.network(noise)


class Discriminator(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=240, output_dim=1):
        super(Discriminator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, output_dim), nn.Sigmoid(),)
    
    def forward(self, x):
        return self.network(x)
    

def criterion(batch_size, device, disc_gen_out, disc_out=None):
    if disc_out is not None:
        #update disc
        neg_loss = torch.nn.BCELoss()(disc_gen_out,
                                        torch.zeros(batch_size, device=device))
        pos_loss = torch.nn.BCELoss()(disc_out,
                                        torch.ones(batch_size, device=device))
        #generator tries to maximise the loss
        #discriminator tries to minimise the loss
        return (neg_loss + pos_loss) / 2
    else:
        return torch.nn.BCELoss()(disc_gen_out,
                                  torch.ones(batch_size, device=device))


def plot_results(noise_samples, generator):
    # NB_IMAGES = 10
    # z = sample_noise(NB_IMAGES).to(device)
    x = generator(noise_samples)

    plt.figure(figsize=(17,17))
    for i in range(noise_samples.shape[0]):
        plt.subplot(5, 5, 1+i)
        plt.axis('off')
        plt.imshow(x[i].data.cpu().numpy().reshape(28,28), cmap='gray')
    plt.show()


# #gen - generator
# #disc - discriminator
# def train(gen, disc, gen_opt, disc_opt, trainX, epochs=25, k=1, batch_size=50):
#     train_loss = {'gen':[],'disc':[]}
#     for epoch in range(epochs):
        
#         #train the disc
#         for _ in range(k):
#             #minibatch of noise (neg labels)
#             z = sample_noise(batch_size).to(device)
#             #minibatch of actual data (pos labels)
#             x = get_minibatch(trainX, batch_size).to(device)
            
#             #update disc
#             neg_loss = torch.nn.BCELoss()(disc(gen(z)).reshape(batch_size),
#                                           torch.zeros(batch_size, device=device))
#             pos_loss = torch.nn.BCELoss()(disc(x).reshape(batch_size),
#                                           torch.ones(batch_size, device=device))
            
#             #generator tries to maximise the loss
#             #discriminator tries to minimise the loss
#             loss = (neg_loss + pos_loss) / 2
            
#             disc_opt.zero_grad()
#             loss.backward()
#             disc_opt.step()
#             train_loss['disc'].append(loss.item())
        
#         #train the gen
#         #minibatch of noise
#         z = sample_noise(batch_size).to(device)
        
#         #update gen
#         #gen tries to maximise
#         #disc tries to minimise
#         loss = torch.nn.BCELoss()(disc(gen(z)).reshape(batch_size),
#                                   torch.ones(batch_size, device=device))
#         gen_opt.zero_grad()
#         loss.backward()
#         gen_opt.step()
#         train_loss['gen'].append(loss.item())
#     return train_loss