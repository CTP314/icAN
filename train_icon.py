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

from model.resnet.gan_resnet import Generator, Discriminator, init_net
    
@dataclass
class TrainConfig:
    device: str = 'cuda:0'
    seed: int = 1
    data_path: str = 'data_new/'
    checkpoints_path: Optional[str] = None # 'ckpt/'  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load

    # 数据集超参数
    themes = None
    bias = True

    # 网络超参数
    num_categories = 10
    input_nc = 8
    output_nc = 4
    ngf = 64
    norm_layer = nn.BatchNorm2d
    use_dropout = False
    n_blocks = 9
    padding_type = 'reflect'
    latent_size = None
    ndf = 128
    n_layers = 3
    init_type = 'kaiming'


    # 训练超参数
    num_epochs: int = 100
    batch_size: int = 64
    eval_freq: int = 1

    # Wandb logging
    is_wandb: bool = False
    project: str = 'icAN'
    name: str = 'resnet-color-ref'

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
        generator,
        generator_optimizer,
        discriminator,
        discriminator_optimizer,
        device,
        L1_penalty=1000, 
        Lconst_penalty=15, 
        Ltv_penalty=0.1,
        Lcategory_penalty=1
    ):
        self.generator = generator
        self.generator_optimizer = generator_optimizer
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.device = device

        self.L1_penalty = L1_penalty
        self.Lconst_penalty = Lconst_penalty
        self.Ltv_penalty = Ltv_penalty
        self.Lcategory_penalty = Lcategory_penalty

        self.total_it = 0

    def train(self, batch):
        self.generator.train()
        self.discriminator.train()

        self.total_it += 1
        icon_ref, icon_src, icon_tar, theme_tar = batch
        log_dict = {}
        # print(icon_src.shape)
        icon_src_enc = self.generator.encoder(torch.cat([icon_src, icon_ref], dim=1))
        icon_fake = self.generator.decoder(icon_src_enc, theme_tar)
        
        real_category_logits, real_D_logits = self.discriminator(icon_tar)
        fake_category_logits, fake_D_logits = self.discriminator(icon_fake)
        
        icon_fake_enc = self.generator.encoder(torch.cat([icon_fake, icon_ref], dim=1))
        const_loss = F.mse_loss(icon_fake_enc, icon_src_enc) * self.Lconst_penalty
        
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
        
        self.generator_optimizer.zero_grad()
        g_loss.backward()
        self.generator_optimizer.step()


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
    
    def generate(self, icon_ref, icon_src, theme_id):
        self.generator.eval()
        with torch.no_grad():
            icon_ref = torch.FloatTensor(icon_ref).to(self.device).permute(2, 0, 1).unsqueeze(0)
            icon_src = torch.FloatTensor(icon_src).to(self.device).permute(2, 0, 1).unsqueeze(0)
            theme_id = torch.LongTensor([theme_id]).to(self.device)
            icon_tar = self.generator(torch.cat([icon_src, icon_ref], 1), theme_id).squeeze().permute(1, 2, 0).cpu().numpy()
        return icon_tar

    def state_dict(self) -> Dict[str, Any]:
        return {
            "encoder": self.generator,
            "discriminator": self.discriminator,
        }
    
    def save(self, path):
        state_dict = {}
        for k, v in self.state_dict().items():
            v.to('cpu')
            state_dict[k] = v.state_dict()
            v.to(self.device)
        torch.save(state_dict, os.path.join(path, f"{self.total_it}.pt"))

    def load_state_dict(self, state_dict: Dict[str, Any]):
        raise NotImplementedError

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
        collate_fn=train_dataset.collate_fn_ab
    )

    seed = config.seed
    set_seed(seed)

    generator = Generator(
        num_categories=config.num_categories,
        input_nc=config.input_nc,
        output_nc=config.output_nc,
        ngf=config.ngf,
        norm_layer=config.norm_layer,
        use_dropout=config.use_dropout,
        n_blocks=config.n_blocks,
        padding_type=config.padding_type,
        latent_size=config.latent_size
    )
    discriminator = Discriminator(
        num_categories=config.num_categories,
        input_nc=config.output_nc,
        ndf=config.ngf,
        n_layers=config.n_layers,
        norm_layer=config.norm_layer   
    )

    init_net(generator, config.init_type, gpu_id=config.device)
    init_net(discriminator, config.init_type, gpu_id=config.device)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4,  betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4,  betas=(0.5, 0.999))

    kwargs = {
        'generator': generator,
        'generator_optimizer': generator_optimizer,
        'discriminator': discriminator,
        'discriminator_optimizer': discriminator_optimizer,
        'device': config.device,
    }

    trainer = ICAN(**kwargs)

    print('-' * 50)
    print(f'Training icAN, Seed: {seed}, Name: {config.name}')
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
            for label in ['app-store', 'chrome', 'weibo', 'genshin-impact', 'adobe-illustrator']:
                output = []
                for theme in train_dataset.themes:
                    if theme in train_dataset.label2theme[label]:
                        icons = [train_dataset.read_icon(label, theme)]
                        for theme_tar in train_dataset.themes:
                            icon_ref = train_dataset.read_icon_with_rlabel(theme_tar, icons[0])
                            icon_tar = trainer.generate(icon_ref, icons[0], theme_id=train_dataset.theme2id[theme_tar])
                            icons.append(icon_ref)
                            icons.append(icon_tar)
                    else:
                        icons = [np.zeros((128, 128, 4))]
                        for theme_tar in train_dataset.themes:
                            icons.append(np.zeros((128, 128, 4)))  
                            icons.append(np.zeros((128, 128, 4)))
                    # print(np.concatenate(icons, axis=1).shape)   
                    output.append(np.concatenate(icons, axis=1))                   

                output = ((np.concatenate(output, axis=0) + 1) / 2 * 255).astype(np.uint8)
                os.makedirs(f'eval/{config.name}/{epoch}', exist_ok=True)
                cv2.imwrite(f'eval/{config.name}/{epoch}/{label}.png', 
                            output, 
                )

            if config.checkpoints_path is not None:
                os.makedirs(config.checkpoints_path, exist_ok=True)
                trainer.save(config.checkpoints_path)

if __name__ == '__main__':
    train()