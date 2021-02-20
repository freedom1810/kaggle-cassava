import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torch
from torch import nn
import math
from copy import deepcopy
from collections import OrderedDict
from sys import stderr
from torch.cuda.amp import autocast
from torch import Tensor

from net.utils import *
from net.ema import *

         
class EfficientNetB3DSPlus(nn.Module):
    def __init__(self, model_params, n_class = 5, pretrained=True):
        super().__init__()
        backbone = timm.create_model(model_params['model_name'], pretrained=pretrained)
        # count_parameters(backbone)
        # ct = 0
        # for name, param in backbone.named_parameters():
        #     if param.requires_grad:
        #         ct +=1 
        #     # if ct <= num_layer - 2:
        #     # print(name)
        # # print('num layer: ',ct)
        # # print(backbone.fc.in_features)
        # print(ct)
        try:
            n_features = backbone.classifier.in_features
        except:
            n_features = backbone.fc.in_features
            #n_features = backbone.head.in_features
        self.backbone = nn.Sequential(*backbone.children())[:-2]
        self.classifier = nn.Linear(n_features, n_class)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward_features(self, x):
        x = self.backbone(x)
        return x

    def forward(self, x, train_state = False):
        # if train_state:
        with autocast(enabled=train_state):
            feats = self.forward_features(x)
            #print('\n1', x.shape, feats.shape, self.classifier)
            x = self.pool(feats).view(x.size(0), -1)
            #print(x.shape)
            x = self.classifier(x)
            #print(x.shape)
        # else:
        #     feats = self.forward_features(x)
        #     x = self.pool(feats).view(x.size(0), -1)
        #     x = self.classifier(x)
        return x, feats
