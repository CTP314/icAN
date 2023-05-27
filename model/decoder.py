import torch
from torch import nn 
import torch.nn.functional as F

class decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.latent_size = config.latent_size
        self.label_size = config.label_size
        self.output_width = config.output_width
        
        self.device = config.device
        self.activation = config.activation
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(self.label_size+self.latent_size, 128, 4, 1, 0),
            nn.BatchNorm2d(128),
            self.activation(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            self.activation(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            self.activation(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            self.activation(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            self.activation(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            self.activation(),
            nn.ConvTranspose2d(32, 4, 3, 1, 1),
        )
        
    def forward(self, latent, label):
        return self.net(torch.cat((latent, label),1))    

class config():
    def __init__(self, latent_size, label_size,  output_width, device, activation):
        self.latent_size = latent_size
        self.output_width = output_width
        self.device = device
        self.activation = activation
        self.label_size =label_size
        
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    
if __name__ == '__main__':
    config_test = config(10, 3, 3, 'cpu', nn.ELU)
    decoder_test = decoder(config_test)
    print_network(decoder_test)
    print(decoder_test(torch.zeros(1,10,1,1), torch.zeros(1,3,1,1)).shape)