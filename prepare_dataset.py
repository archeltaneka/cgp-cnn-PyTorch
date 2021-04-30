import numpy as np
import pandas as pd
import os

import torch
from torch.utils.data import Dataset
import torchvision
from sklearn.model_selection import train_test_split

from PIL import Image

class CoronaHackDataset(Dataset):
    
    def __init__(self, metadata, root_dir, transform=None):
        self.root_dir = root_dir
        self.metadata = metadata
        self.transform = transform
        self.list_dir = os.listdir(root_dir)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        metadata = self.metadata.iloc[idx]
        fname = metadata['X_ray_image_name']
        
        try:
            file_idx = self.list_dir.index(fname)
        except:
            print('Image not found!')
            return None
        
        img_path = os.path.join(self.root_dir, self.list_dir[file_idx])
        img = Image.open(img_path).convert('RGB')
        img = img.resize((128,128))
        tensor_img = self.transform(img)
        tensor_lbl = torch.tensor(metadata['target'].item())
        
        return tensor_img, tensor_lbl

def prepare(data_dir):
    # CoronaHack dataset
    img_path = data_dir + 'Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/'
    train_dir = img_path + 'train'
    test_dir = img_path + 'test'

    metadata = pd.read_csv(os.path.join(data_dir, 'Chest_xray_Corona_Metadata.csv'))
    df_train = metadata[metadata['Dataset_type']=='TRAIN']
    df_test = metadata[metadata['Dataset_type']=='TEST']

    df_train.loc[df_train['Label'].eq('Normal'), 'class'] = 'healthy'
    df_train.loc[(df_train['class'].ne('healthy') & df_train['Label_1_Virus_category'].eq('bacteria')), 'class'] = 'COVID-19';
    df_train.loc[(df_train['class'].ne('healthy') & df_train['class'].ne('bacteria') & df_train['Label_2_Virus_category'].eq('COVID-19')), 'class'] = 'COVID-19';
    df_train.loc[(df_train['class'].ne('healthy') & df_train['class'].ne('bacteria') & df_train['class'].ne('COVID-19')), 'class'] = 'COVID-19';

    df_test.loc[df_test['Label'].eq('Normal'), 'class'] = 'healthy'
    df_test.loc[(df_test['class'].ne('healthy') & df_test['Label_1_Virus_category'].eq('bacteria')), 'class'] = 'COVID-19';
    df_test.loc[(df_test['class'].ne('healthy') & df_test['class'].ne('bacteria') & df_test['Label_2_Virus_category'].eq('COVID-19')), 'class'] = 'COVID-19';
    df_test.loc[(df_test['class'].ne('healthy') & df_test['class'].ne('bacteria') & df_test['class'].ne('COVID-19')), 'class'] = 'COVID-19';

    target_dict = {'healthy': 0, 'COVID-19': 1}
    df_train['target'] = df_train['class'].map(target_dict)
    df_test['target'] = df_test['class'].map(target_dict)

    train_ds, val_ds = train_test_split(df_train, test_size=0.12, random_state=1, shuffle=True)
    train_ds, val_ds = train_ds.reset_index(drop=True), val_ds.reset_index(drop=True)
    
    batch_size = 32
    normalize = False
    stats = ((0.0093, 0.0093, 0.0092),(0.4827, 0.4828, 0.4828))

    if normalize:
        train_tfms = torchvision.transforms.ToTensor()
    else:
        train_tfms = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(128), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(*stats)])
    test_tfms = test_tfms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(*stats)])
    
    train_dataset = CoronaHackDataset(train_ds, train_dir, transform=train_tfms)
    valid_dataset = CoronaHackDataset(val_ds, train_dir, transform=test_tfms)
    
    return train_dataset, valid_dataset