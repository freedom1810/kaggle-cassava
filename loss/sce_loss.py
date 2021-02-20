import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

class SCELoss(torch.nn.Module):
    def __init__(self, alpha=0.1, beta=1, num_classes=5):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels, TTA = False):
        labels = labels.to(dtype=torch.long)
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        #print('pred_TTA:', pred.shape)
        if not TTA:
            pred = F.softmax(pred, dim=1)
            #print('pred:', pred.shape)

        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss