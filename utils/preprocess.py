from tqdm import tqdm
import cv2
import os
import numpy as np
from PIL import Image
root_path = 'data/'
data_path = root_path + 'raw'
edge_path = root_path + 'edge'
meta_path = root_path + 'meta'

themes = []
labels = []
theme2label = {}
label2theme = {}
loader = [(a, b, c) for a, b, c in os.walk(data_path)]
with tqdm(loader, desc='loading dataset...') as pbar:
    for filepath, dirnames, filenames in pbar:
        if len(filepath.split('/')) <= 2:
            continue
        label = filepath.split('/')[-1]
        # print(filepath, filenames)
        labels.append(label)
        label2theme[label] = []
        os.makedirs(os.path.join(edge_path, 'edge', label), exist_ok=True)
        for theme_file in filenames:
            if theme_file[-3:] == 'png':
                theme = theme_file[:-4]
                # print(os.path.join(self.data_path, label, theme+'.png'))
                img = cv2.imread(os.path.join(data_path, label, theme+'.png'), cv2.IMREAD_UNCHANGED)
                # print(type(img), os.path.join(data_path, label, theme+'.png'))
                if img is not None and img.shape == (128, 128, 4):
                    label2theme[label].append(theme)
                    img = cv2.Canny(img, 10, 100)
                    cv2.imwrite(os.path.join(edge_path, label, theme+'_edge.png'), img)
                    if theme not in themes:
                        themes.append(theme)
                        theme2label[theme] = []
                    
                    theme2label[theme].append(label)
            # else:
            #     try:
            #         print(f'remove {label} {theme}')
            #         os.rmdir(os.path.join(data_path, label, theme+'.png'))
            #     except:
            #         pass

headers = []
srcs = []

for label in labels:
    for theme_src in label2theme[label]:
        headers.append(f'{label}-{theme_src}')
        srcs.append(cv2.imread(os.path.join(edge_path, label, theme_src+'_edge.png')))

tars = {}
rows = []
for theme_tar in tqdm(themes, leave=False):
    tars[theme_tar] = []
    for label_ref in tqdm(theme2label[theme_tar]):
        tars[theme_tar].append(cv2.imread(os.path.join(edge_path, label_ref, theme_tar+'_edge.png')))

for label in tqdm(labels, leave=False):
    os.makedirs(os.path.join(meta_path, label), exist_ok=True)
    for theme_src in tqdm(label2theme[label], leave=False):
        src = cv2.imread(os.path.join(edge_path, label, theme_src+'_edge.png'))
        for theme_tar in themes:
            dis = [np.abs(src - tar).sum() for tar in tars[theme_tar]]
            inds = np.argsort(dis)[:10]
            names = [theme2label[theme_tar][i] for i in inds]
            file = open(os.path.join(meta_path, label, f'{theme_src}-{theme_tar}.txt'), 'w')
            mid = str(names).replace('[', '').replace(']', '')
            # 删除单引号并用字符空格代替逗号
            mid = mid.replace("'", '').replace(',', '') + '\n'
            file.write(mid)
            file.close()
