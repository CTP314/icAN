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
from train_icon_only_ref import ICAN, TrainConfig

def f2uint(img):
    return ((img + 1) / 2 * 255).astype(np.uint8)

if __name__ == '__main__':
    config = TrainConfig()
    data_path = './data'
    device = 'cuda'
    model_path = 'ckpt/resnet-color-ref-a1542418/27000.pt'
    dataset = IconDataset(data_path, device)

    state_dict = torch.load(model_path)
    # print(state_dict.keys())
    # icAN = ICAN()

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

    generator.load_state_dict(state_dict['encoder'])
    discriminator.load_state_dict(state_dict['discriminator'])

    print(dataset.themes)

    icon = dataset.read_icon('chrome', 'office40')