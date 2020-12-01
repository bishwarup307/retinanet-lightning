"""
__author__: bishwarup307
Created: 25/11/20
"""
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Implements Focal loss for dense object detection (https://arxiv.org/abs/1708.02002)
    """

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

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        num_pos: torch.Tensor,
        num_assigned: torch.Tensor,
    ):
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * labels + (1 - p) * (1 - labels)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha = self.alpha * labels + (1.0 - self.alpha) * (1.0 - labels)
            loss = alpha * loss

        loss = loss.sum(axis=1)
        scatter_indices = torch.arange(len(num_assigned)).type_as(num_assigned)
        scatter_indices = torch.repeat_interleave(scatter_indices, num_assigned)

        image_losses = torch.zeros_like(num_assigned).float()
        image_losses = image_losses.scatter_add(0, scatter_indices, loss)

        num_pos = torch.clamp(num_pos, min=1.0)
        image_losses = image_losses / num_pos
        return image_losses.mean()

        # if self.reduction == "mean":
        #     loss = loss.mean()
        #
        # if self.reduction == "sum":
        #     loss = loss.sum()
        #
        # return loss
