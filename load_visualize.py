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
import matplotlib.pyplot as plt
from canny_with_grad.net_canny import Net
from imageio import imread, imsave

from dataset import IconDataset

from model.resnet.gan_resnet import Generator, Discriminator, init_net

# input shape = bsz * cnl * w * h
def compute_pixel_color_distribution(batched_image, real=True):
    distribution_dict = []
    print('start!!!!!!!')
    for image in batched_image:
        distribution = {}
        for w in range(image.shape[1]):
            for h in range(image.shape[2]):
                key = str(np.round(image[:,w, h].detach().cpu().numpy(), 1))
                distribution[key] = distribution[key] + 1 if key in distribution else 1        
        distribution_dict.append(distribution)
        print(sorted(distribution.values(), reverse=True))
        cumulative = np.array([np.array(sorted(distribution.values(), reverse=True)[:i+1]).sum()/w/h for i in range(30)])
        if real :
            cumulative_label = 'cumulative real'
            color = 'color real'
            real = False
        else:
            cumulative_label = 'cumulative fake'
            color = 'color fake'
        plt.plot(np.log2(cumulative+1e-5)*10, label=cumulative_label)
        plt.plot(np.log2(np.array(sorted(distribution.values(), reverse=True)[1:31])/sorted(distribution.values(), reverse=True)[0]+1e-5), label = color)
        plt.legend()
        plt.savefig("image"+str(random.random())+".png")
        print(len(distribution.values()))
        return 
    return distribution_dict
    
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
    num_epochs: int = 200
    batch_size: int = 6
    eval_freq: int = 10

    # Wandb logging
    is_wandb: bool = True
    project: str = 'icAN'
    name: str = 'resnet-color-ref'

def load(name: str, config=TrainConfig):
    dictionary = torch.load('ckpt/resnet-color-ref-'+name+"/27000.pt")
    
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
    
    generator.load_state_dict(dictionary["encoder"])
    discriminator.load_state_dict(dictionary['discriminator'])
    
    generator.to(config.device)
    discriminator.to(config.device)
    
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
    
    for data in train_loader:
        icon_ref, icon_src, icon_tar, theme_tar = data
        print('start')
        compute_pixel_color_distribution(icon_tar)
        icon_src_enc = generator.encoder(torch.cat([icon_src, icon_ref], dim=1))
        icon_fake = generator.decoder(icon_src_enc, theme_tar)
        compute_pixel_color_distribution(icon_fake, real=False)
        return 0
    
if __name__ == '__main__':
    load("8e40fddf")