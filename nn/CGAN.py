"""
This file is an implementation of the pix2pix-Model from:
Image-to-Image Translation with Conditional Adversarial Networks
https://arxiv.org/abs/1611.07004
"""


import torch
import torch.nn as nn
from tqdm import tqdm
#from utils import sample_noise


class Block(nn.Module):
    def __init__(self, in_chs, out_chs, dec=False):
        super().__init__()
        if dec:
            self.c = nn.ConvTranspose2d(in_channels=in_chs,out_channels=out_chs,kernel_size=4,stride=2,padding=1)    
        else:
            self.c = nn.Conv2d(in_channels=in_chs,out_channels=out_chs,kernel_size=4,stride=2,padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_chs)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.c(x)))


#Generator
class Generator(nn.Module):
    def __init__(self, inp_chs=3):
        super().__init__()
        self.encoder = nn.Sequential(
            Block(in_chs=inp_chs,out_chs=64),
            Block(in_chs=64,out_chs=128),
            Block(in_chs=128,out_chs=256),
            Block(in_chs=256,out_chs=512),
            Block(in_chs=512,out_chs=512),
            Block(in_chs=512,out_chs=512),
            Block(in_chs=512,out_chs=512),
            Block(in_chs=512,out_chs=512),
        )
        
        self.decoder = nn.Sequential(
            Block(in_chs=512,out_chs=512,dec=True),
            Block(in_chs=512,out_chs=512,dec=True),
            Block(in_chs=512,out_chs=512,dec=True),
            Block(in_chs=512,out_chs=512,dec=True),
            Block(in_chs=512,out_chs=256,dec=True),
            Block(in_chs=256,out_chs=128,dec=True),
            Block(in_chs=128,out_chs=64,dec=True),
        )
        self.proj = nn.ConvTranspose2d(in_channels=64,out_channels=inp_chs,kernel_size=4,stride=2,padding=1)
        
    def forward(self, x, z):
        #also should add some noise
        return torch.tanh(self.proj(self.decoder(self.encoder(x))))




#Discriminator
class Discriminator(nn.Module):
    def __init__(self, inp_chs=6, out_chs=(64,128)):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=inp_chs,out_channels=64,kernel_size=4,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),             
        )
        self.pred = nn.Linear(in_features=128*126*126,out_features=1)
        
    def forward(self, x, y):
        x = torch.cat((x,y),dim=1) # concat along channel
        x = self.discriminator(x)
        x = x.view(x.size(0),-1) # Flatten the vector
        return torch.sigmoid(self.pred(x))


class CGAN(nn.Module): #Conditional Gan
    def __init__(self, batch_size, inp_chs_gen=3, inp_chs_dis=6, out_chs_dis=(64,128), device='cpu') -> None:
        super().__init__()
        self.generator = Generator(inp_chs=inp_chs_gen)
        self.discriminator = Discriminator(inp_chs=inp_chs_dis,out_chs=out_chs_dis)
        self.batch_size = batch_size
        self.device = device
    
    def criterion(self, disc_out, disc_gen_out=None):
        if disc_out is not None:
            f_loss = torch.nn.BCELoss()(disc_gen_out,
                                            torch.zeros(self.batch_size, device=self.device))
            r_loss = torch.nn.BCELoss()(disc_out, torch.ones(self.batch_size, device=self.device))
            return (r_loss + f_loss) / 2
        else:
            return torch.nn.BCELoss()(disc_gen_out, torch.ones(self.batch_size, device=self.device))
            
    
    def forward(self, noise, batch, gen_step=False): #batch is (x,y) tuple of imgs
        if gen_step:
            return self.discriminator(batch[1],self.generator(batch[0],noise))
        return self.discriminator(batch[1],self.generator(batch[0],noise)), self.discriminator(batch[0],batch[1])
        




# def train(generator, discriminator, generator_optimizer, discriminator_optimizer, nb_epochs, k=1, batch_size=100, WGAN=False):
#     training_loss = {'generative': [], 'discriminator': []}
    
#     for epoch in tqdm(range(nb_epochs)):
#         # Train the discriminator
#         for _ in range(k):
#             # Sample a minibatch of m noise samples
#             z = sample_noise(batch_size).to(device)
#             # Sample a minibatch of m examples from the data generating distribution
#             x,y = get_minibatch(batch_size)
#             x = x.to(device)
#             y = y.to(device)
            

#             # Update the discriminator by ascending its stochastic gradient
#             f_loss = torch.nn.BCELoss()(discriminator(x, generator(x, z)).reshape(batch_size),
#                                         torch.zeros(batch_size, device=device))
#             r_loss = torch.nn.BCELoss()(discriminator(x, y).reshape(batch_size), torch.ones(batch_size, device=device))
#             loss = (r_loss + f_loss) / 2
#             discriminator_optimizer.zero_grad()
#             loss.backward()
#             discriminator_optimizer.step()
            
#             training_loss['discriminator'].append(loss.item())

#         # Train the generator
#         # Sample a minibatch of m noise samples
#         z = sample_noise(batch_size).to(device)
#         x,y = get_minibatch(batch_size).to(device)
#         # Update the generator by descending its stochastic gradient
#         loss = torch.nn.BCELoss()(discriminator(y, generator(x, z)).reshape(batch_size), torch.ones(batch_size, device=device))
#         generator_optimizer.zero_grad()
#         loss.backward()
#         generator_optimizer.step()
#         training_loss['generative'].append(loss.item())

#     return training_loss


# # g = Generator()
# # d = Discriminator()
# # x = torch.randn(1,3,512,512)
# # z = sample_noise(1)


# # y = g(x, z) # out of gen
# # print(x.shape, y.shape)
# # o = d(x,y) # out of dis

# # print("gen out shape ", y.shape)
# # print(o)






# #TODO:
# #add noise input




# # x = torch.randn(1,3,512,512)
# # c = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=4,stride=2,padding=1)
# # c1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1)
# # print(c1(c(x)).shape)

# from torch import optim
# import matplotlib.pyplot as plt

# #Training loop
# device = 'cpu'

# discriminator = Discriminator().to(device)
# generator = Generator().to(device)

# optimizer_d = optim.SGD(discriminator.parameters(), lr=0.1, momentum=0.5)
# optimizer_g = optim.SGD(generator.parameters(), lr=0.1, momentum=0.5)

# loss = train(generator, discriminator, optimizer_g, optimizer_d, nb_epochs=1, batch_size=100, WGAN=False)

# # Sample and plot images from the trained generator
# # NB_IMAGES = 25
# # z = sample_noise(NB_IMAGES).to(device)
# # x = generator(z)
# # plt.figure(figsize=(17, 17))
# # for i in range(NB_IMAGES):
# #     plt.subplot(5, 5, 1 + i)
# #     plt.axis('off')
# #     plt.imshow(x[i].data.cpu().numpy().reshape(28, 28), cmap='gray')
# # plt.savefig("Imgs/regenerated_MNIST_data.png")
# # plt.show()