import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torchcontrib.optim import SWA
from albumentations.pytorch import ToTensorV2
import math
from torch.nn import BCELoss
import pandas as pd
from sklearn.utils import shuffle

path_params = {
    'csv_path': "dataset/train_mix/new_mix_v3_fuse2_fold13.csv",
    # 'csv_path': "/home/hana/sonnh/kaggle-cassava/dataset/train_mix/new_mix_v3.csv",
    'img_path': "dataset/original_mix/",
    'save_path': "checkpoints/76/{}_fold-{}"

}

model_params = {
    'model_name': 'tf_efficientnet_b4_ns',
    # 'model_name': 'seresnext50_32x4d',
    #'model_name': 'ViT-B_32',
    #'model_name': 'vit_base_patch32_384',
    'img_size': [512, 512],
    'num_classes': 5,
    'ds': False,
    'ds_blocks': [10, 15],
    'special_augment_prob': 1.,
    'EMA': 1,
    'EMA_model': ''
    
}

optimizer_params = {
    'lr': 5e-4
    'weighted_loss': True,
    # 'weight_loss': torch.tensor([1.0, 0.43, 0.49, 0.1, 0.52]),
    'weight_loss': None,
    'lrf': 1e-2
}

training_params = {
    'training_batch_size': 30,
    'num_workers': 16,
    'device': torch.device("cuda:0"),
    'device_ids': [0, 1],
    'start_epoch': 1,
    'num_epoch': 50,
    'warm_up': 0,
    'TTA_time': 1
}


# config = {

#     'path_params' : {
#         'csv_path': "/home/hana/sonnh/kaggle-cassava/dataset/train_mix/new_mix_v3_fuse2_fold13.csv",
#         # 'csv_path': "/home/hana/sonnh/kaggle-cassava/dataset/train_mix/new_mix_v3.csv",
#         'img_path': "/home/hana/sonnh/kaggle-cassava/dataset/original_mix/",
#         'save_path': "checkpoints/75/{}_fold-{}"

#     },

#     'model_params' : {
#         'model_name': 'tf_efficientnet_b4_ns',
#         # 'model_name': 'seresnext50_32x4d',
#         #'model_name': 'ViT-B_32',
#         #'model_name': 'vit_base_patch32_384',
#         'img_size': [512, 512],
#         'num_classes': 5,
#         'ds': False,
#         'ds_blocks': [10, 15],
#         'special_augment_prob': 1.,
#         'EMA': 1,
#         'EMA_model': ''
        
#     },

#     'optimizer_params' : {
#         'weighted_loss': True,
#         # 'weight_loss': torch.tensor([1.0, 0.43, 0.49, 0.1, 0.52]),
#         'weight_loss': None,
#         'lrf': 1e-2
#     },

#     'training_params' : {
#         'training_batch_size': 30,
#         'num_workers': 16,
#         'device': torch.device("cuda:0"),
#         'device_ids': [0, 1],
#         'start_epoch': 1,
#         'num_epoch': 50,
#         'warm_up': 5,
#         'TTA_time': 5
#     }

# }
