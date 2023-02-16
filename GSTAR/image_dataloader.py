from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.utils as vutils
import torch
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import PIL



class CelebADataset(Dataset):
    def __init__(self, path, transform=None):

        self.data_csv = pd.read_csv(path, header = None)
        
        self.sens = self.data_csv[0].apply(lambda x: x.split(' ')[-1]).apply(int) #sensitive attribute
        self.label = self.data_csv[0].apply(lambda x: x.split(' ')[-2]).apply(int) #target label
        self.img = self.data_csv[0].apply(lambda x: x.split(' ')[0]) #image
        
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img[idx]
        image = PIL.Image.open(img_name)
        
        a = self.sens[idx]
        y = self.label[idx]

        if self.transform:
            image = self.transform(image)

        return image, a, y
    
def show_img(img):
    plt.figure(figsize = (10,10))
    plt.imshow(np.transpose(vutils.make_grid(img.cpu()), (1,2,0)))