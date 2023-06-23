import cv2
import os
from tqdm import tqdm

data_path = 'data'
loader = [(a, b, c) for a, b, c in os.walk(data_path)]
with tqdm(loader, desc='loading dataset...', leave=False) as pbar:
    for filepath, dirnames, filenames in pbar:
        label = filepath[5:]
        for theme_file in filenames:
            theme = theme_file[:-4]
            # print(os.path.join(self.data_path, label, theme+'.png'))
            img = cv2.imread(os.path.join(data_path, label, theme+'.png'), cv2.IMREAD_UNCHANGED)
            if img is None or img.shape != (128, 128, 4):
                print(label, theme)
                os.remove(os.path.join(data_path, label, theme+'.png'))