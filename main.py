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
from albumentations import *
import gc

from config import *

from loss.focal_cosine_loss import FocalCosineLoss
from loss.label_smoothing_loss import LabelSmoothingLoss

from net.nets import EfficientNetB3DSPlus
from net.ema import ModelEMA 
from net.utils import count_parameters

from dataloader.data import CassaDataset

from engine.engines_fp16 import trainer_augment

from optimizer.sam import SAM


df = pd.read_csv(path_params['csv_path'])

for fold in [1,2,3,4,5]:
    """StratifiedKFold"""
    print("="*20, "Fold", fold, "="*20)
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    train_df = shuffle(train_df, random_state = 2020)
    eval_df = df[df["fold"] == fold].reset_index(drop=True)
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
            df=train_df[:100],
            image_folder=path_params['img_path'],
            image_transform=train_transform,
        ), 
        batch_size=training_params['training_batch_size'], 
        num_workers=training_params['num_workers'],
        shuffle=True
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
            df=eval_df[:100],
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
    # ct = 0
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         ct +=1 
    # print(ct)

    if model_params['EMA']:
        model_params['ema_model'] = ModelEMA(model)
    if optimizer_params['weighted_loss']:
        # criterion = SCELoss()
        criterion = FocalCosineLoss()
        val_criterion = LabelSmoothingLoss(smoothing = 0, weight = optimizer_params['weight_loss'], training = False)

    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    lf = lambda x: ((1 + math.cos(x * math.pi / training_params['num_epoch'])) / 2) * (1 - optimizer_params['lrf']) + optimizer_params['lrf']  # cosine
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    trainer_augment(loaders, 
                    model_params, 
                    model, 
                    criterion, 
                    val_criterion, 
                    optimizer, 
                    lr_scheduler, 
                    optimizer_params, 
                    training_params, 
                    save_path= path_params['save_path'].format(model_params['model_name'], fold))

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()