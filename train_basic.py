import numpy as np
import pyrallis
import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution
import torch.nn as nn
import torch.nn.functional as F
import wandb
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass
import os
import random
import uuid
import cv2
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import IconDataset

from model.basic.encoder import Encoder, Discriminator
from model.basic.decoder import Decoder
    
@dataclass
class TrainConfig:
    device: str = 'cuda:0'
    seed: int = 1
    data_path: str = 'data/'
    checkpoints_path: Optional[str] = 'ckpt/'  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load

    # 数据集超参数
    themes = None
    bias = True

    # 网络超参数
    latent_size: int = 1024
    label_size: int = 128
    image_width: int = 128
    activation = nn.ELU
    kernel_channels: int = 64
    image_channels: int = 4
    num_categories: int = 270 # 270
    L1_penalty = 1000 
    Lconst_penalty = 15 
    Ltv_penalty = 0.1
    Lcategory_penalty = 1    

    # 训练超参数
    num_epochs: int = 50
    batch_size: int = 128
    eval_freq: int = 10

    # Wandb logging
    is_wandb: bool = True
    project: str = 'icAN'
    name: str = 'basic'

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

def set_seed(
    seed: int, deterministic_torch: bool = False
):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)   

def wandb_init(config: dict):
    wandb.init(
        config=config,
        project=config['project'],
        name=config['name'],
        id=str(uuid.uuid4())
    )
    wandb.run.save()

class ICAN:
    def __init__(
        self,
        encoder,
        encoder_optimizer,
        decoder,
        decoder_optimizer,
        discriminator,
        discriminator_optimizer,
        L1_penalty=1000, 
        Lconst_penalty=15, 
        Ltv_penalty=0.1,
        Lcategory_penalty=1
    ):
        self.encoder = encoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder = decoder
        self.decoder_optimizer = decoder_optimizer
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer


        self.L1_penalty = L1_penalty
        self.Lconst_penalty = Lconst_penalty
        self.Ltv_penalty = Ltv_penalty
        self.Lcategory_penalty = Lcategory_penalty

        self.total_it = 0

    def train(self, batch):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

        self.total_it += 1
        icon_src, icon_tar, theme_tar = batch
        log_dict = {}
        # print(icon_src.shape)
        icon_src_enc = self.encoder(icon_src)
        icon_fake = self.decoder(icon_src_enc, theme_tar)
        
        real_category_logits, real_D_logits = self.discriminator(icon_tar)
        fake_category_logits, fake_D_logits = self.discriminator(icon_fake)
        
        icon_fake_enc = self.encoder(icon_fake)
        const_loss = F.mse_loss(icon_fake_enc, icon_src_enc) * self.Lconst_penalty
        # const_loss = torch.zeros(1).to(self.encoder.device)
        
        real_category_loss = F.cross_entropy(real_category_logits, theme_tar)
        fake_category_loss = F.cross_entropy(fake_category_logits, theme_tar)
        category_loss = self.Lcategory_penalty * (real_category_loss + fake_category_loss)

        d_loss_real = F.binary_cross_entropy_with_logits(real_D_logits, torch.ones_like(real_D_logits, dtype=torch.float32))
        d_loss_fake = F.binary_cross_entropy_with_logits(fake_D_logits, torch.zeros_like(fake_D_logits, dtype=torch.float32))

        l1_loss = F.l1_loss(icon_fake, icon_tar) * self.L1_penalty

        width = icon_fake.size(2)
        # print(icon_fake.shape)
        tv_loss = (F.mse_loss(icon_fake[..., 1:, :], icon_fake[..., :width - 1, :]) / width
               + F.mse_loss(icon_fake[..., 1:], icon_fake[..., :width - 1]) / width) * self.Ltv_penalty
        
        cheat_loss = F.binary_cross_entropy_with_logits(fake_D_logits, torch.ones_like(fake_D_logits, dtype=torch.float32))

        g_loss = cheat_loss + l1_loss + self.Lcategory_penalty * fake_category_loss + const_loss + tv_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        g_loss.backward()
        
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()


        real_category_logits, real_D_logits = self.discriminator(icon_tar.detach())
        fake_category_logits, fake_D_logits = self.discriminator(icon_fake.detach())
        
        real_category_loss = F.cross_entropy(real_category_logits, theme_tar)
        fake_category_loss = F.cross_entropy(fake_category_logits, theme_tar)
        category_loss = self.Lcategory_penalty * (real_category_loss + fake_category_loss)

        d_loss_real = F.binary_cross_entropy_with_logits(real_D_logits, torch.ones_like(real_D_logits, dtype=torch.float32))
        d_loss_fake = F.binary_cross_entropy_with_logits(fake_D_logits, torch.zeros_like(fake_D_logits, dtype=torch.float32))


        d_loss = d_loss_real + d_loss_fake + category_loss / 2.0
        self.discriminator_optimizer.zero_grad()
        d_loss.backward()
        self.discriminator_optimizer.step()

        log_dict['const_loss'] = const_loss.detach().cpu().numpy()
        log_dict['real_category_loss'] = real_category_loss.detach().cpu().numpy()
        log_dict['fake_category_loss'] = fake_category_loss.detach().cpu().numpy()
        log_dict['category_loss'] = category_loss.detach().cpu().numpy()
        log_dict['l1_loss'] = l1_loss.detach().cpu().numpy()
        log_dict['tv_loss'] = tv_loss.detach().cpu().numpy()
        log_dict['cheat_loss'] = cheat_loss.detach().cpu().numpy()
        log_dict['d_loss'] = d_loss.detach().cpu().numpy()
        log_dict['g_loss'] = g_loss.detach().cpu().numpy()
        
        return log_dict
    
    def generate(self, icon_src, theme_id):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            icon_src = torch.FloatTensor(icon_src).to(self.encoder.device).permute(2, 0, 1).unsqueeze(0)
            theme_id = torch.LongTensor([theme_id]).to(self.encoder.device)
            icon_src_enc = self.encoder(icon_src)
            icon_tar = self.decoder(icon_src_enc, theme_id).squeeze(0).detach().permute(1, 2, 0).cpu().numpy()
            icon_tar = (icon_tar * 255).astype(np.uint8)
        return icon_tar

    def state_dict(self) -> Dict[str, Any]:
        return {
            "encoder": self.encoder,
            "decoder": self.decoder,
            "discriminator": self.discriminator,
        }
    
    def save(self, path):
        for k, v in self.state_dict().items():
            v.to('cpu')
            torch.save(v.state_dict(), os.path.join(path, f"{k}_{self.total_it}.pt"))  
            v.to('cuda')

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.encoder_optimizer.load_state_dict(state_dict["encoder_optimizer"])

        self.decoder.load_state_dict(state_dict["decoder"])
        self.decoder_optimizer.load_state_dict(state_dict["decoder_optimizer"])

        self.discriminator.load_state_dict(state_dict["discriminator"])
        self.discriminator_optimizer.load_state_dict(state_dict["discriminator_optimizer"])

        self.total_it = state_dict["total_it"]

@pyrallis.wrap()
def train(config: TrainConfig):
    train_dataset = IconDataset(
        data_path=config.data_path, 
        device=config.device,
        themes=config.themes,
        bias=config.bias
    )
    train_loader = DataLoader(
        train_dataset, 
        config.batch_size, 
        shuffle=True, 
        collate_fn=train_dataset.collate_fn
    )

    seed = config.seed
    set_seed(seed)

    encoder = Encoder(config).to(config.device)
    discriminator = Discriminator(config).to(config.device)
    decoder = Decoder(config).to(config.device)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=2e-4,  betas=(0.5, 0.999))
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=2e-4,  betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4,  betas=(0.5, 0.999))

    kwargs = {
        'encoder': encoder,
        'encoder_optimizer': encoder_optimizer,
        'decoder': decoder,
        'decoder_optimizer': decoder_optimizer,
        'discriminator': discriminator,
        'discriminator_optimizer': discriminator_optimizer
    }

    trainer = ICAN(**kwargs)

    print('-' * 50)
    print(f'Training icAN, Seed: {seed}')
    print('-' * 50)

    if config.load_model != '':
        raise NotImplementedError
    
    if config.is_wandb:
        wandb_init(asdict(config))

    for epoch in range(config.num_epochs):
        with tqdm(train_loader) as pbar:
            for batch in pbar:
                log_dict = trainer.train(batch)

                pbar.set_description(
                    'Epoch: %d, D_loss: %.4f, G_loss: %.4f'%(epoch, log_dict['d_loss'], log_dict['g_loss'])
                )

                if config.is_wandb:
                    wandb.log(log_dict)

        if (epoch + 1) % config.eval_freq == 0:
            print(f'Epoch: {epoch} evaluating...')
            for theme in train_dataset.themes:
                for label in ['app-store', 'chrome', 'weibo', 'genshin-impact']:
                    icon_src = train_dataset.read_icon(label, 'ios7')
                    icon_tar = trainer.generate(icon_src, theme_id=train_dataset.theme2id[theme])
                    os.makedirs(f'eval/{config.name}/{epoch}/{label}', exist_ok=True)
                    cv2.imwrite(f'eval/{config.name}/{epoch}/{label}/{theme}.png', 
                                icon_tar, 
                                [cv2.IMWRITE_PNG_COMPRESSION, 0]
                    )
                    
            if config.checkpoints_path is not None:
                os.makedirs(config.checkpoints_path, exist_ok=True)
                trainer.save(config.checkpoints_path)

if __name__ == '__main__':
    train()