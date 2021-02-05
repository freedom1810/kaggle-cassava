import torch
import torch.nn as nn
import torch.nn.functional as F
# from efficientnet_pytorch import EfficientNet
import timm
import torch
from torch import nn
import math
from copy import deepcopy
from collections import OrderedDict
from sys import stderr
from torch.cuda.amp import autocast
# for type hint
from torch import Tensor

from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print("Total Trainable Params: {}".format(total_params))
    return total_params

def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def copy_attr(a, b, include=(), exclude=()):          
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)
            
class EfficientNetB3DSPlus(nn.Module):
    def __init__(self, model_params, n_class = 5, pretrained=True):
        super().__init__()
        backbone = timm.create_model(model_params['model_name'], pretrained=pretrained)
        # count_parameters(backbone)
        # for name, param in backbone.named_modules():
        #     # if param.requires_grad:
        #     # ct +=1 
        #     # if ct <= num_layer - 2:
        #     print(name)
        # # print(ct)
        # print(backbone.fc.in_features)

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


def init_linears(classifier):
    linears = [module for module in classifier.modules() if isinstance(module, nn.Linear)]
    for i, module in enumerate(linears):
        if module.bias is not None:
            nn.init.constant_(module.bias, val=0)

        if i < len(linears) - 1:
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
        else:
            nn.init.xavier_uniform_(module.weight)

class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

    
class EfficientNetB3DSPlus_old(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.effn = EfficientNet.from_pretrained(model_params['model_name'], advprop=True)
        self.model_params = model_params

        self.aux0_linear = nn.Sequential(
            nn.Linear(self.effn._blocks[self.model_params["ds_blocks"][0]]._bn2.num_features, 
            self.model_params["num_classes"]))
        init_linears(self.aux0_linear)
        '''
        self.aux1_linear = nn.Sequential(
            nn.Linear(self.effn._blocks[self.model_params["ds_blocks"][1]]._bn2.num_features, 
            self.model_params["num_classes"]))
        init_linears(self.aux1_linear)
        '''
        self.head_linear = nn.Sequential(
            nn.Linear(self.effn._fc.in_features, self.model_params["num_classes"]))
        init_linears(self.head_linear)

    def subforward(self, image):
        deep_features = self.effn._bn0(self.effn._conv_stem(image))
        for i in range(len(self.effn._blocks)):
            deep_features = self.effn._blocks[i](deep_features)
            
            if i == self.model_params["ds_blocks"][0]:
                aux0_features = deep_features
            '''
            if i == self.model_params["ds_blocks"][1]:
                aux1_features = deep_features
            '''
        head_features = self.effn._swish(self.effn._bn1(self.effn._conv_head(deep_features)))

        #return aux0_features, aux1_features, head_features
        return aux0_features, head_features

    def forward(self, image):
        '''
        aux0_features, aux1_features, head_features = self.subforward(image)

        aux0_features = F.adaptive_avg_pool2d(aux0_features, 1)
        aux0_features = aux0_features.view(aux0_features.size(0), -1)
        aux0_features = self.aux0_linear(aux0_features)
        aux1_features = F.adaptive_avg_pool2d(aux1_features, 1)
        aux1_features = aux1_features.view(aux1_features.size(0), -1)
        aux1_features = self.aux1_linear(aux1_features)
        '''
        aux0_features, head_features_ori = self.subforward(image)
        aux0_features = F.adaptive_avg_pool2d(aux0_features, 1)
        aux0_features = aux0_features.view(aux0_features.size(0), -1)
        aux0_features = self.aux0_linear(aux0_features)

        head_features = F.adaptive_avg_pool2d(head_features_ori, 1)
        head_features = head_features.view(head_features.size(0), -1)
        head_features = self.head_linear(head_features)

        #avg_features = (aux0_features + aux1_features + head_features)/3

        #output = self.classifier(avg_features)

        return head_features_ori, head_features, aux0_features