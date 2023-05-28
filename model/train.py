import numpy as np
import torch
from .encoder import encoder, discriminator
from .decoder import decoder
from .embedding import get_embedding

seed = 123123

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

config = None
dataloader = None

L1_penalty = config.L1_penalty
Lconst_penalty = config.Lconst_penalty
Ltv_penalty = config.Ltv_penalty
Lcategory_penalty = config.Lcategory_penalty

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

epochs = config.epochs
encoder_net_1 = encoder(config)
encoder_net_2 = encoder(config)
discriminator_net = discriminator(config)
decoder_net = decoder(config)

optimizer_e1 = torch.optim.Adam(encoder_net_1.parameters(), lr=2e-4,  betas=(0.5, 0.999))
optimizer_e2 = torch.optim.Adam(encoder_net_2.parameters(), lr=2e-4,  betas=(0.5, 0.999))
optimizer_decoder = torch.optim.Adam(decoder_net.parameters(), lr=2e-4,  betas=(0.5, 0.999))
optimizer_discriminator = torch.optim.Adam(discriminator_net.parameters(), lr=2e-4,  betas=(0.5, 0.999))

for epoch in range(epochs):
    for data in dataloader:
        input_image, labels, target,  = data
        image_embedding = encoder_net_1(image_embedding)
        label_embedding = get_embedding(labels)
        new_image = decoder(image_embedding, label_embedding)
        
        
        