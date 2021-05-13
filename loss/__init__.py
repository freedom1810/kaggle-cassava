from loss.focal_cosine_loss import FocalCosineLoss
from loss.label_smoothing_loss import LabelSmoothingLoss
from loss.bi_tempered_logistic_loss import BTLLoss
from loss.sce_loss import SCELoss
from torch.nn import BCELoss

LOSSES = {'FocalCosineLoss': FocalCosineLoss,
          'LabelSmoothingLoss': LabelSmoothingLoss,
          'bi_tempered_logistic_loss': BTLLoss,
          'SCELoss': SCELoss,
          'BCELoss': BCELoss}
