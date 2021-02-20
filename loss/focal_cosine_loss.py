import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, xent=.1):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.xent = xent

        self.y = torch.Tensor([1]).cuda()

    def forward(self, input, target, reduction="mean"):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target.long(), num_classes=input.size(-1)), self.y, reduction=reduction)
        # print(target.size())
        # print(input.size())
        # cosine_loss = F.cosine_embedding_loss(input, target, self.y, reduction=reduction)

        cent_loss = F.cross_entropy(F.normalize(input), target.long(), reduce=False)

        # cosine_loss = F.cosine_embedding_loss(F.normalize(input), F.one_hot(target, num_classes=input.size(-1)), self.y, reduction=reduction)
        # cent_loss = F.cross_entropy(input, target, reduce=False)

        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * cent_loss

        if reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss