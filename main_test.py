import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
import albumentations.pytorch as AT
from sklearn.model_selection import StratifiedKFold
from data import CassaDataset
from nets import EfficientNetB3DSPlus, ModelEMA
from losses import LabelSmoothingLoss
from engines_test import trainer_augment
from torchcontrib.optim import SWA
#from one_cycle import OneCycleLR
from torch.nn import BCELoss
import pandas as pd
from sklearn.utils import shuffle
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
import math

df = pd.read_csv("/home/hana/sonnh/kaggle-cassava/dataset/train_mix/mix.csv")


#from losses import LabelSmoothingCrossEntropy

#skf = StratifiedKFold(n_splits=5, random_state=2020)
#skf.get_n_splits(df, df["target"].values)

model_params = {
    'model_name': 'tf_efficientnet_b0_ns',
    'img_size': [512, 512],
    "fuse_linear": 512,
    'num_classes': 5,
    'ds_blocks': [10, 15],
    'weight': torch.tensor([1.0, 0.43, 0.49, 0.1, 0.52]),
    'weight': None,
    'special_augment_prob': 1.,
    'device_ids': [0, 1]
}

for fold in range(1):
    """StratifiedKFold"""
    print("="*20, "Fold", fold + 1, "="*20)
    train_df = df[df["fold"] % 5 != fold].reset_index(drop=True)
    train_df = shuffle(train_df, random_state = 2020)
    eval_df = df[df["fold"] % 5 == fold].reset_index(drop=True)
    eval_df = shuffle(eval_df, random_state = 2020)
    """==============="""
    training_batch_size = 80
    num_workers = 10
    train_transform = A.Compose([
            RandomResizedCrop(model_params['img_size'][0], model_params['img_size'][1]),
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
            df=train_df[:1000],
            image_folder="/home/hana/sonnh/kaggle-cassava/dataset/train_mix/image/", 
            image_transform=train_transform,
        ), 
        batch_size=training_batch_size, 
        num_workers=num_workers,
        #sampler=BalanceClassSampler(list(train_df["target"].values), "downsampling"),
    )

    eval_transform = A.Compose([
        #A.Resize(448, 448),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0)
    ])
    eval_loader = torch.utils.data.DataLoader(
        CassaDataset(
            df=eval_df[:100], 
            #meta_columns=list(eval_df.columns)[1:-2],
            image_folder="/home/hana/sonnh/kaggle-cassava/dataset/train_mix/image/", 
            image_transform=eval_transform,
        ), 
        batch_size=training_batch_size*2, 
        num_workers=num_workers,
    )

    loaders = {
        "train": train_loader,
        "eval": eval_loader,
    }
    model = EfficientNetB3DSPlus(model_params).to(torch.device("cuda:0"))
    #EMA = model()
    ema = ModelEMA(model)
    criterion = LabelSmoothingLoss(weight = model_params['weight'])
    #criterion = LabelSmoothingCrossEntropy()
    num_epoch = 50
    
    optimizer = optim.Adam(model.parameters(), lr=5*1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1/10**0.5)
    #lf = lambda x: ((1 + math.cos(x * math.pi / num_epoch)) / 2) * (1 - 2*10e-3) + 2*10e-3  # cosine
    #lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    #optimizer = SWA(optimizer)
    '''
    MAX_LR = 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR)
    lr_scheduler = OneCycleLR(optimizer, max_lr=MAX_LR,
		steps_per_epoch=len(train_loader), epochs=num_epoch)
    optimizer = SWA(optimizer)
    '''
    trainer_augment(loaders, model_params['special_augment_prob'], model, criterion, optimizer, lr_scheduler, 
                    start_epoch=0, total_epochs=num_epoch, 
                    device=torch.device("cuda:0"), device_ids = model_params['device_ids'],
                    save_path= "checkpoints/{}_fold-{}_test2.pt".format(model_params['model_name'], fold + 1),
                    ema = ema)