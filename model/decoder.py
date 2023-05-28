import torch
from torch import nn 
import torch.nn.functional as F

class decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.latent_size = config.latent_size
        self.label_size = config.label_size
        self.kernel_channels = config.kernel_channels
        self.image_width = config.image_width
        self.image_channels = config.image_channels
        
        self.device = config.device
        self.activation = config.activation
    
    # decoder current setting: input latent variable and style embedding
    # output bsz * channels * height * width
    
        self.net = nn.Sequential(
            nn.ConvTranspose2d(self.label_size+self.latent_size, 8 * self.kernel_channels, self.image_width//32, 1, 0),
            self.activation(),
            nn.ConvTranspose2d(8 * self.kernel_channels, 4 * self.kernel_channels, 4, 2, 1),
            nn.BatchNorm2d(4 * self.kernel_channels),
            self.activation(),
            nn.ConvTranspose2d(4 * self.kernel_channels, 2 * self.kernel_channels, 4, 2, 1),
            nn.BatchNorm2d(2 * self.kernel_channels),
            self.activation(),
            nn.ConvTranspose2d(2 * self.kernel_channels, self.kernel_channels, 4, 2, 1),
            nn.BatchNorm2d(self.kernel_channels),
            self.activation(),
            nn.ConvTranspose2d(self.kernel_channels, self.kernel_channels, 4, 2, 1),
            nn.BatchNorm2d(self.kernel_channels),
            self.activation(),
            nn.ConvTranspose2d(self.kernel_channels, self.kernel_channels, 4, 2, 1),
            nn.BatchNorm2d(self.kernel_channels),
            self.activation(),
            nn.ConvTranspose2d(self.kernel_channels, self.image_channels, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, latent, label):
        return self.net(torch.cat((latent, label),1))    

class config():
    def __init__(self, latent_size, label_size,  output_width, device, activation,kernel_channels, output_channels):
        self.latent_size = latent_size
        self.image_width = output_width
        self.device = device
        self.activation = activation
        self.label_size = label_size
        self.kernel_channels = kernel_channels
        self.image_channels = output_channels
        
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    
if __name__ == '__main__':
    config_test = config(10, 3, 128, 'cpu', nn.ELU, 64, 4)
    decoder_test = decoder(config_test)
    print_network(decoder_test)
    print(decoder_test(torch.zeros(1,10,1,1), torch.zeros(1,3,1,1)).shape)