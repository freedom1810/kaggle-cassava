import json
import tqdm
import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics as metrics
from torch.cuda import amp
import torch.nn.functional as F


def freeze_model(model, num_layer):
    ct = 0
    for param in model.parameters():
        ct += 1
        if ct <= num_layer - 2:
            param.requires_grad = False

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    return model

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(True)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(True)
            module.train()

    return model

def save_checkpoint(model_eval, optimizer, lr_scheduler, epoch, epoch_save_path):
    state_dicts = {
            "model_state_dict": model_eval.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "start_epoch": epoch + 1
        }

    torch.save(state_dicts, epoch_save_path)