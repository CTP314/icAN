import torch
from torch import nn 
import torch.nn.functional as F

class Encoder(nn.Module):
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
            nn.Conv2d(self.image_channels, self.kernel_channels, 3, 1, 1),
            self.activation(),
            nn.BatchNorm2d(self.kernel_channels),
            nn.Conv2d(self.kernel_channels, self.kernel_channels, 4, 2, 1),
            self.activation(),
            nn.BatchNorm2d(self.kernel_channels),
            nn.Conv2d(self.kernel_channels, self.kernel_channels, 4, 2, 1),
            self.activation(),
            nn.BatchNorm2d(self.kernel_channels),
            nn.Conv2d(self.kernel_channels, self.kernel_channels * 2, 4, 2, 1),
            self.activation(),
            nn.BatchNorm2d(self.kernel_channels * 2),
            nn.Conv2d(self.kernel_channels * 2 , self.kernel_channels * 4, 4, 2, 1),
            self.activation(),
            nn.BatchNorm2d(self.kernel_channels * 4),
            nn.Conv2d(self.kernel_channels * 4 , self.kernel_channels * 8, 4, 2, 1),
            self.activation(),
            nn.BatchNorm2d(self.kernel_channels * 8),
            nn.Conv2d(self.kernel_channels * 8, self.latent_size, self.image_width//32, 1, 0),
            self.activation(),
        )
        
    def forward(self, image):
        return self.net(image)    

class config():
    def __init__(self, latent_size, label_size,  output_width, device, activation,kernel_channels, output_channels, num_categories):
        self.latent_size = latent_size
        self.image_width = output_width
        self.device = device
        self.activation = activation
        self.label_size = label_size
        self.kernel_channels = kernel_channels
        self.image_channels = output_channels
        self.num_categories = num_categories
        
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

class Discriminator(Encoder):
    def __init__(self, config):
        super().__init__(config)
        self.latent_size = config.latent_size
        self.num_categories = config.num_categories
        self.category_discriminator = nn.Linear(self.latent_size, self.num_categories)
        self.true_false_discriminator = nn.Linear(self.latent_size, 1)
    
    def forward(self, image):
        representations = super().forward(image)
        # print(representations.shape)
        representations = representations.squeeze()
        # print(representations.shape)
        return self.category_discriminator(representations), self.true_false_discriminator(representations)
        
if __name__ == '__main__':
    config_test = config(10, 3, 128, 'cpu', nn.ELU, 64, 4, 10)
    encoder_test = Encoder(config_test)
    discriminator_test = Discriminator(config_test)
    print_network(encoder_test)
    print(encoder_test(torch.zeros(1,4,128,128)).shape)
    print(discriminator_test((torch.zeros(3,4,128,128))))