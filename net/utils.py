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
