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
        self.root_path = data_path
        self.data_path = data_path + '/raw'
        self.edge_path = data_path + '/edge'
        self.meta_path = data_path + '/meta'
        self.themes = []
        labels = []
        self.theme2label = {}
        self.label2theme = {}
        loader = [(a, b, c) for a, b, c in os.walk(self.data_path)]
        with tqdm(loader, desc='loading dataset...') as pbar:
            for filepath, dirnames, filenames in pbar:
                if len(filepath.split('/')) <= 2:
                    continue
                label = filepath.split('/')[-1]
                labels.append(label)
                self.label2theme[label] = []
                for theme_file in filenames:
                    if theme_file[-3:] == 'png':
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
        return {
            'img': icon,
            'label': label,
            'theme': theme,
        }
    
    def read_icon_edge(self, label, theme):
        icon = cv2.imread(
            os.path.join(self.edge_path, label, theme+'_edge.png'), 
            cv2.IMREAD_UNCHANGED, 
        ).astype(np.float64) / 255 * 2 - 1
        return {
            'img': icon,
            'label': label,
            'theme': theme,
            'edge': True,
        }
    
    def __getitem__(self, index):
        label, theme = self.data[index]
        # print(label, theme)
        icon = self.read_icon(label, theme)
        # assert theme in ['ios11', 'ios7']
        return icon
    
    def read_icon_with_rtheme(self, label):
        t = np.random.choice(self.label2theme[label])
        return self.read_icon(label, t)
    
    def read_icon_with_rlabel(self, theme, icon):
        with open(os.path.join(self.meta_path, icon['label'], icon['theme'] + '-' + theme + '.txt'), 'r') as f:
            line = f.readline().strip().split(' ')
        label_ref = np.random.choice(line)
        try:
            icon = self.read_icon(label_ref, theme)
        except:
            icon = {'img': np.zeros((128, 128, 4))}
        return icon
    
    def collate_fn(self, samples):
        icons_S = []
        icons_T = []
        themes_T = []
        for info in samples:
            icon, theme, label = info['img'], info['theme'], info['label']
            icon_S = torch.FloatTensor(self.read_icon_with_rtheme(label)['img']).to(self.device).permute(2, 0, 1).unsqueeze(0)
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
        for info in samples:
            icon, theme, label = info['img'], info['theme'], info['label']
            icon_S = self.read_icon_with_rtheme(label)
            icon_ref = torch.FloatTensor(self.read_icon_with_rlabel(theme, icon_S)['img']).to(self.device).permute(2, 0, 1).unsqueeze(0)
            icon_S = torch.FloatTensor(icon_S['img']).to(self.device).permute(2, 0, 1).unsqueeze(0)
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
    icon = train_data[0]
    print(icon['img'].shape, icon['label'], icon['theme'])

    train_dataloader = DataLoader(train_data, 32, collate_fn=train_data.collate_fn)
    
    print(len(train_dataloader))
    icon_edge = train_data.read_icon_edge('app-store', 'clouds')
    train_data.read_icon_with_rlabel(icon['theme'], icon_edge)
    print(icon_edge['img'].shape)