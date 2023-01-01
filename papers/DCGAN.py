"""
This file is an implementation of the PSPNet-Model from:
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
https://arxiv.org/abs/1511.06434
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from keras.datasets.mnist import load_data



#load (and normalize) mnist dataset
(trainX, trainy), (testX, testy) = load_data()
trainX = (np.float32(trainX) - 127.5) / 127.5


def get_minibatch(batch_size):
    indices = torch.randperm(trainX.shape[0])[:batch_size]
    return nn.functional.interpolate(torch.tensor(trainX[indices], dtype=torch.float).reshape(batch_size, 1, 28, 28),size=(109,109))


def sample_noise(size, dim=100):
    out = torch.empty(size, dim)
    mean = torch.zeros(size, dim)
    std = torch.ones(dim)
    torch.normal(mean, std, out=out)
    return out



class Generator(nn.Module):
    def __init__(self, input_dim=100):
        super(Generator, self).__init__()
        
        self.proj = nn.Linear(in_features=input_dim,out_features=1024*4*4)
        #reshape to tensor (1024,4,4)
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=5,stride=2),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=5,stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=5,stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,out_channels=1,kernel_size=5,stride=2),
            nn.BatchNorm2d(num_features=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        img = self.proj(noise).view(-1,1024,4,4)
        return self.conv_blocks(img)
    


class Discriminator(nn.Module):
    def __init__(self, inp_chs=3, output_dim=1):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=inp_chs,out_channels=64,kernel_size=5,stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=5,stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=5,stride=2),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(in_features=4*4*512,out_features=output_dim),
            nn.Sigmoid(),)

    def forward(self, x):
        return self.network(x)


def train(generator, discriminator, generator_optimizer, discriminator_optimizer, nb_epochs, k=1, batch_size=100, WGAN=False):
    training_loss = {'generative': [], 'discriminator': []}
    for epoch in tqdm(range(nb_epochs)):

        # Train the disciminator
        for _ in range(k):
            # Sample a minibatch of m noise samples
            z = sample_noise(batch_size).to(device)
            # Sample a minibatch of m examples from the data generating distribution
            x = get_minibatch(batch_size).to(device)

            # Update the discriminator by ascending its stochastic gradient
            #WGAN loss
            if WGAN:
                loss = -(torch.mean(discriminator(x).reshape(batch_size)) - torch.mean(discriminator(generator(z)).reshape(batch_size)))
            else:
                f_loss = torch.nn.BCELoss()(discriminator(generator(z)).reshape(batch_size),
                                            torch.zeros(batch_size, device=device))
                r_loss = torch.nn.BCELoss()(discriminator(x).reshape(batch_size), torch.ones(batch_size, device=device))
                loss = (r_loss + f_loss) / 2
            discriminator_optimizer.zero_grad()
            loss.backward()
            discriminator_optimizer.step()
            
            for p in discriminator.parameters():
                p.data.clamp_(-0.01,0.01)
            
            training_loss['discriminator'].append(loss.item())

        # Train the generator
        # Sample a minibatch of m noise samples
        z = sample_noise(batch_size).to(device)
        # Update the generator by descending its stochastic gradient
        #WGAN loss
        if WGAN:
            loss = -torch.mean(discriminator(generator(z)).reshape(batch_size))
        else:
            loss = torch.nn.BCELoss()(discriminator(generator(z)).reshape(batch_size),
                                    torch.ones(batch_size, device=device))
        generator_optimizer.zero_grad()
        loss.backward()
        generator_optimizer.step()
        training_loss['generative'].append(loss.item())

    return training_loss


if __name__ == "__main__":
    gen = Generator()
    dis = Discriminator(inp_chs=1)
    
    z = sample_noise(1)
    d = nn.functional.interpolate(get_minibatch(3),size=(109,109))
    print(d.shape)
    x = gen(z)
    print(x.shape)
    print(dis(d))
    print(dis(d).reshape(3))
    
    
    
    
    # device = 'cpu'

    # discriminator = Discriminator(inp_chs=1).to(device)
    # generator = Generator().to(device)

    # optimizer_d = optim.SGD(discriminator.parameters(), lr=0.1, momentum=0.5)
    # optimizer_g = optim.SGD(generator.parameters(), lr=0.1, momentum=0.5)

    # loss = train(generator, discriminator, optimizer_g, optimizer_d, 10, batch_size=100, WGAN=True)

    # # Sample and plot images from the trained generator
    # NB_IMAGES = 25
    # z = sample_noise(NB_IMAGES).to(device)
    # x = generator(z)
    # plt.figure(figsize=(17, 17))
    # for i in range(NB_IMAGES):
    #     plt.subplot(5, 5, 1 + i)
    #     plt.axis('off')
    #     plt.imshow(x[i].data.cpu().numpy().reshape(28, 28), cmap='gray')
    # plt.savefig("Imgs/regenerated_MNIST_data.png")
    # plt.show()