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

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.2, reduction="mean", weight=None, training=True):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = smoothing
        self.reduction = reduction
        self.weight = weight
        self.training = training

    def reduce_loss(self, loss):
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def linear_combination(self, x, y):
        #print(x, y, self.epsilon)
        return self.epsilon * x + (1 - self.epsilon) * y

    def forward(self, preds, target, TTA = False):
        target = target.to(dtype=torch.long)
        target = torch.squeeze(target)
        if self.weight is not None:
            self.weight = self.weight.to(preds.device)
        #print('---------------------')
        #print(target.shape, preds.shape)
        if self.training:
            #print('Train')
            n = preds.size(-1)
            log_preds = F.log_softmax(preds, dim=-1)
            loss = self.reduce_loss(-log_preds.sum(dim=-1))
            #print(log_preds.shape)
            nll = F.nll_loss(log_preds, target, reduction=self.reduction, weight=self.weight)
            return self.linear_combination(loss / n, nll)
        else:
            if not TTA:
                log_preds = F.log_softmax(preds, dim=-1)
            else:
                log_preds = preds
            return torch.nn.functional.cross_entropy(preds, target, weight=self.weight)


def log_t(u, t):
    """Compute log_t for `u`."""

    if t == 1.0:
        return torch.log(u)
    else:
        return (u ** (1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
    """Compute exp_t for `u`."""

    if t == 1.0:
        return torch.exp(u)
    else:
        return torch.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))


def compute_normalization_fixed_point(activations, t, num_iters=5):
    """Returns the normalization value for each example (t > 1.0).
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (> 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    mu = torch.max(activations, dim=-1).values.view(-1, 1)
    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0
    i = 0
    while i < num_iters:
        i += 1
        logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)
        normalized_activations = normalized_activations_step_0 * (logt_partition ** (1.0 - t))

    logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)

    return -log_t(1.0 / logt_partition, t) + mu


def compute_normalization(activations, t, num_iters=5):
    """Returns the normalization value for each example.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    if t < 1.0:
        return None # not implemented as these values do not occur in the authors experiments...
    else:
        return compute_normalization_fixed_point(activations, t, num_iters)


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature tensor > 0.0.
    num_iters: Number of iterations to run the method.
    Returns:
    A probabilities tensor.
    """

    if t == 1.0:
        normalization_constants = torch.log(torch.sum(torch.exp(activations), dim=-1))
    else:
        normalization_constants = compute_normalization(activations, t, num_iters)

    return exp_t(activations - normalization_constants, t)


def bi_tempered_logistic_loss(activations, labels, t1 = 0.6, t2 = 1.2, label_smoothing=0.1, num_iters=5):
    labels = labels.to(int)
    labels = torch.nn.functional.one_hot(labels, num_classes = 5)
    """Bi-Tempered Logistic Loss with custom gradient.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    labels: A tensor with shape and dtype as activations.
    t1: Temperature 1 (< 1.0 for boundedness).
    t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    label_smoothing: Label smoothing parameter between [0, 1).
    num_iters: Number of iterations to run the method.
    Returns:
    A loss tensor.
    """

    if label_smoothing > 0.0:
        num_classes = labels.shape[-1]
        labels = (1 - num_classes / (num_classes - 1) * label_smoothing) * labels + label_smoothing / (num_classes - 1)

    probabilities = tempered_softmax(activations, t2, num_iters)
    #print('\n', probabilities.shape, labels.shape)
    temp1 = (log_t(labels + 1e-10, t1) - log_t(probabilities, t1)) * labels
    temp2 = (1 / (2 - t1)) * (torch.pow(labels, 2 - t1) - torch.pow(probabilities, 2 - t1))
    loss_values = temp1 - temp2

    return torch.mean(torch.sum(loss_values, dim=-1))

    # loss = torch.mean(torch.sum(loss_values, dim=-1))
    # return torch.minimum(loss, torch.ones_like(loss)*0.75)


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