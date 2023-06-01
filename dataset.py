import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import cv2

class IconDataset(Dataset):
    def __init__(self, data_path, device, themes=None, bias=True) -> None:
        super(IconDataset, self).__init__()
        self.device = device
        self.data_path = data_path

        self.themes = []
        labels = []
        self.theme2label = {}
        self.label2theme = {}
        loader = [(a, b, c) for a, b, c in os.walk(data_path)]
        with tqdm(loader, desc='loading dataset...') as pbar:
            for filepath, dirnames, filenames in pbar:
                label = filepath.split('/')[1]
                labels.append(label)
                self.label2theme[label] = []
                for theme_file in filenames:
                    theme = theme_file[:-4]
                    # print(os.path.join(self.data_path, label, theme+'.png'))
                    img = cv2.imread(os.path.join(self.data_path, label, theme+'.png'), cv2.IMREAD_UNCHANGED)
                    if img is not None and img.shape == (128, 128, 4):
                        self.label2theme[label].append(theme)

                        if theme not in self.themes:
                            self.themes.append(theme)
                            self.theme2label[theme] = []
                        
                        self.theme2label[theme].append(label)

        if themes is not None:
            self.themes = themes

        if not bias:
            labels_ = []
            for theme in self.themes:
                labels_.append(set(self.theme2label[theme]))
            
            self.labels = list(set.intersection(*labels_))
            for theme in self.themes:
                self.theme2label[theme] = self.labels   
            for label in self.labels:
                self.label2theme[label] = self.themes
        else:
            self.labels = labels

        self.label2id = {f: k for k, f in enumerate(self.labels)}
        self.theme2id = {f: k for k, f in enumerate(self.themes)}

        self.data = []
        self.shape = (len(self.labels), len(self.themes))

        for theme in self.themes:
            for label in self.theme2label[theme]:
                self.data.append((label, theme))

        print('-' * 50)
        print(f'#icons: {len(self.data)}, #themes: {len(self.themes)}')

    def __len__(self):
        return len(self.data)
    
    def read_icon(self, label, theme):
        icon = cv2.imread(
            os.path.join(self.data_path, label, theme+'.png'), 
            cv2.IMREAD_UNCHANGED, 
        ).astype(np.float64) / 255 * 2 - 1
        return icon
    
    def read_icon_edge(self, label, theme):
        icon = self.read_icon(label, theme)
        return cv2.Canny(icon, 10, 100)[..., None]
    
    def __getitem__(self, index):
        label, theme = self.data[index]
        # print(label, theme)
        icon = self.read_icon(label, theme)
        # assert theme in ['ios11', 'ios7']
        return icon, label, theme
    
    def read_icon_with_rtheme(self, label):
        t = np.random.choice(self.label2theme[label])
        return self.read_icon(label, t)
    
    def read_icon_with_rlabel(self, theme, icon):
        labels = np.random.choice(self.theme2label[theme], 100)
        l_ref = labels[0]
        dis_ref = np.abs(self.read_icon(l_ref, theme) - icon).sum()
        for l in labels:
            dis = np.abs(self.read_icon(l, theme) - icon).sum()
            if dis_ref > dis:
                l_ref = l
                dis_ref = dis
        return self.read_icon(l_ref, theme)
    
    def collate_fn(self, samples):
        icons_S = []
        icons_T = []
        themes_T = []
        for icon, label, theme in samples:
            icon_S = torch.FloatTensor(self.read_icon_with_rtheme(label)).to(self.device).permute(2, 0, 1).unsqueeze(0)
            assert icon_S.size(2) == 128
            icon_T = torch.FloatTensor(icon).to(self.device).permute(2, 0, 1).unsqueeze(0)
            assert theme in self.themes
            assert theme in self.theme2id.keys()
            theme_T = torch.LongTensor([self.theme2id[theme]]).to(self.device)
            icons_S.append(icon_S)
            icons_T.append(icon_T)
            themes_T.append(theme_T)

        icons_S = torch.concatenate(icons_S, dim=0)
        icons_T = torch.concatenate(icons_T, dim=0)
        themes_T = torch.concatenate(themes_T, dim=0)

        return icons_S, icons_T, themes_T
    
    def collate_fn_ab(self, samples):
        icons_ref = []
        icons_S = []
        icons_T = []
        themes_T = []
        for icon, label, theme in samples:
            icon_S = self.read_icon_with_rtheme(label)
            icon_ref = torch.FloatTensor(self.read_icon_with_rlabel(theme, icon_S)).to(self.device).permute(2, 0, 1).unsqueeze(0)
            icon_S = torch.FloatTensor(icon_S).to(self.device).permute(2, 0, 1).unsqueeze(0)
            assert icon_S.size(2) == 128
            icon_T = torch.FloatTensor(icon).to(self.device).permute(2, 0, 1).unsqueeze(0)
            assert theme in self.themes
            assert theme in self.theme2id.keys()
            theme_T = torch.LongTensor([self.theme2id[theme]]).to(self.device)
            icons_ref.append(icon_ref)
            icons_S.append(icon_S)
            icons_T.append(icon_T)
            themes_T.append(theme_T)

        icons_ref = torch.concatenate(icons_ref, dim=0)
        icons_S = torch.concatenate(icons_S, dim=0)
        icons_T = torch.concatenate(icons_T, dim=0)
        themes_T = torch.concatenate(themes_T, dim=0)

        return icons_ref, icons_S, icons_T, themes_T

            

if __name__ == '__main__':
    # train_data = IconDataset('data/', device='cuda', themes=['win10', 'ios11'], bias=False)
    train_data = IconDataset('data/', device='cuda')
    print(len(train_data))
    icon, label, theme = train_data[0]
    print(icon.shape, label, theme)

    train_dataloader = DataLoader(train_data, 32, collate_fn=train_data.collate_fn)

    print(len(train_dataloader))
    icon_edge = train_data.read_icon_edge('app-store', 'clouds')
    print(icon_edge.shape)
    cv2.imwrite('edge.png', icon_edge)
    with tqdm(train_dataloader, desc='loading...') as pbar:
        for icons_S, icons_T, themes_T in pbar:
            # print(icons_S.shape, icons_T.shape, themes_T.shape)
            pass
    # print('=' * 50)


# dribbble 3d-plastilina
# zalo