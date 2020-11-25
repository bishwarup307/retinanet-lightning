"""
__author__: bishwarup307
Created: 25/11/20
"""
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[float] = 0.25,
        gamma: Optional[float] = 2.0,
        reduction: str = "none",
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # mask = labels > -1
        # masked_lables = labels[mask]
        # masked_logits = logits[mask]
        #
        # masked_labels_one_hot = F.one_hot(
        #     masked_lables, num_classes=logits.size(-1) + 1
        # )[:, 1:].float()
        # no loss for background class
        # (https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/meta_arch/retinanet.py#L321)

        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * labels + (1 - p) * (1 - labels)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha = self.alpha * labels + (1.0 - self.alpha) * (1.0 - labels)
            loss = alpha * loss

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss
