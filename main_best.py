import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from data import CassaDataset
from nets_best import EfficientNetB3DSPlus, ModelEMA
from losses import LabelSmoothingLoss, SCELoss
from engines_best import trainer_augment
from torchcontrib.optim import SWA
from albumentations.pytorch import ToTensorV2
import math
from torch.nn import BCELoss
import pandas as pd
from sklearn.utils import shuffle
from albumentations import (
    RandomCrop,
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)



path_params = {
    'csv_path': "/home/hana/sonnh/kaggle-cassava/dataset/train_mix/mix.csv",
    'img_path': "/home/hana/sonnh/kaggle-cassava/dataset/original_mix/",
    'save_path': "checkpoints/50/{}_fold-{}"

}

model_params = {
    'model_name': 'tf_efficientnet_b4_ns',
    'img_size': [512, 512],
    'num_classes': 5,
    'ds': False,
    'ds_blocks': [10, 15],
    'special_augment_prob': 1.,
    'EMA': 1,
    'EMA_model': ''
    
}

optimizer_params = {
    'weighted_loss': True,
    'weight_loss': None,
    'lrf': 2e-3
}

training_params = {
    'training_batch_size': 30,
    'num_workers': 10,
    'device': torch.device("cuda:0"),
    'device_ids': [0, 1],
    'start_epoch': 1,
    'num_epoch': 50,
    'warm_up': 10,
    'TTA_time': 5
}

df = pd.read_csv(path_params['csv_path'])

for fold in range(5):
    """StratifiedKFold"""
    print("="*20, "Fold", fold + 1, "="*20)
    train_df = df[df["fold"] % 5 != fold].reset_index(drop=True)
    train_df = shuffle(train_df, random_state = 2020)
    eval_df = df[df["fold"] % 5 == fold].reset_index(drop=True)
    eval_df = shuffle(eval_df, random_state = 2020)
    """==============="""
    
    train_transform = Compose([
            RandomResizedCrop(model_params['img_size'][0], model_params['img_size'][1], interpolation = cv2.INTER_CUBIC),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)

    train_loader = torch.utils.data.DataLoader(
        CassaDataset(
            df=train_df[:],
            image_folder=path_params['img_path'],
            image_transform=train_transform,
        ), 
        batch_size=training_params['training_batch_size'], 
        num_workers=training_params['num_workers']
    )

    eval_transform = Compose([
        CenterCrop(600, 600),
        Resize(model_params['img_size'][0], model_params['img_size'][1], cv2.INTER_AREA),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.5),
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ])
    eval_loader = torch.utils.data.DataLoader(
        CassaDataset(
            df=eval_df[:],
            image_folder=path_params['img_path'], 
            image_transform=eval_transform,
        ), 
        batch_size=training_params['training_batch_size']*2, 
        num_workers=training_params['num_workers']
    )

    loaders = {
        "train": train_loader,
        "eval": eval_loader,
    }
    model = EfficientNetB3DSPlus(model_params).to(training_params['device'])
    if model_params['EMA']:
        model_params['ema_model'] = ModelEMA(model)
    if optimizer_params['weighted_loss']:
        criterion = LabelSmoothingLoss(smoothing = 0.2, weight = optimizer_params['weight_loss'], training = True)
        val_criterion = LabelSmoothingLoss(smoothing = 0, weight = optimizer_params['weight_loss'], training = False)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=1)

    trainer_augment(loaders, model_params, model, criterion, val_criterion, optimizer, lr_scheduler, 
                    optimizer_params, training_params, 
                    save_path= path_params['save_path'].format(model_params['model_name'], fold + 1))
    del model, optimizer