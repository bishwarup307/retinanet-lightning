"""
__author__: bishwarup307
Created: 21/11/20
"""
from typing import Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn

from retinanet.utils import ifnone, to_tensor


class Defaults:
    pyramid_levels = [3, 4, 5, 6, 7]
    scales = [2 ** (x / 3.0) for x in torch.arange(3)]
    ratios = [0.5, 1.0, 2.0]
    sizes = [32, 64, 128, 256, 512]
    strides = [8, 16, 32, 64, 128]  # resnet strides + two additional downsamples
    prior_variance = [0.1, 0.1, 0.2, 0.2]


class MultiBoxPrior(nn.Module):
    """
    Generates anchors (priors) across all pyramid levels for multiscale object detection.
    https://arxiv.org/abs/1708.02002
    Args:
        pyramid_levels (Optional[Sequence[int]]): levels of backbone network for which priors
        should be calculated. Defaults to [3, 4, 5, 6, 7] as described in the paper.
        strides (Optional[Sequence[int]]): strides (downsample ratio) of the feature maps at
        specified pyramid levels. Defaults to [8, 16, 32, 64, 128].
        sizes (Optional[Sequence[int]]): base size of anchors on top of which `scales` and
        `ratios` are imposed. Defaults to [32^2, 64^2, 128^2, 256^2, 512^2] as described in the
        paper.
        scales (Optional[Sequence[int]]): scales for the anchors. Defaults to [1, 2 ^ (1/3), 2 ^ (2/3)]
        as described in the paper.
        ratios (Optional[Sequence[int]]): aspect ratios for the anchors. Defaults to [0.5, 1, 2] as
        described in the paper.

    Examples:
        >> m = MultiBoxPrior()
        >> sample_input = torch.randn(4, 3, 256, 256)
        >> anchors = m(sample_input)
    """

    def __init__(
        self,
        pyramid_levels: Optional[Sequence[int]] = None,
        strides: Optional[Sequence[int]] = None,
        sizes: Optional[Sequence[int]] = None,
        scales: Optional[Sequence[int]] = None,
        ratios: Optional[Sequence[int]] = None,
    ):
        super(MultiBoxPrior, self).__init__()
        self.pyramid_levels = ifnone(pyramid_levels, Defaults.pyramid_levels)
        self.strides = ifnone(strides, Defaults.strides)
        self.sizes = ifnone(sizes, Defaults.sizes)
        self.scales = to_tensor(ifnone(scales, Defaults.scales))
        self.ratios = to_tensor(ifnone(ratios, Defaults.ratios))

    @property
    def num_anchors(self):
        return len(self.scales) * len(self.ratios)

    @torch.no_grad()
    def forward(self, image):
        im_hw = np.array(image.size()[2:])
        #         fmap_sizes = [tuple(im_hw // x) for x in self.strides]
        fmap_sizes = [tuple(np.ceil(im_hw / x).astype(int)) for x in self.strides]

        all_anchors = []
        for (level, size, stride, fmap_size) in zip(self.pyramid_levels, self.sizes, self.strides, fmap_sizes):
            anchors = _generate_anchors(size, self.scales, self.ratios)
            anchors = _project_anchors(fmap_size, stride, anchors)
            all_anchors.append(anchors)

        all_anchors = torch.cat(all_anchors)
        return all_anchors


def _generate_anchors(size: int, scales: torch.Tensor, ratios: torch.Tensor) -> torch.Tensor:
    """
    calculates N = S x A anchors given S scales and A aspect ratios (AR).
    Args:
        size [int]: base size of the anchors on which scales and AR will be applied.
        In retinanet implementation, the authors consider the sizes (32**2, 64**2, 128**2,
        256**2, 512**2)
        scales [torch.Tensor]: scales of the anchors
        ratios [torch.Tensor]: aspect ratios for the anchors

    Returns:
        [torch.Tensor]: tensor of size (N, 4). the anchors are in the format (x_min, y_min, x_max, y_max)
    """
    # scales = _get_default_scales() if scales is None else scales
    # ratios = _get_default_ratios() if ratios is None else ratios

    num_anchors = len(scales) * len(ratios)
    anchors = torch.zeros(num_anchors, 4)

    wh = scales.repeat(2 * len(ratios)).view(2, -1).T * size
    areas = torch.prod(wh, dim=1)

    ratios = ratios.repeat(len(scales), 1).T.flatten()
    anchors[:, 3] = torch.sqrt(areas / ratios)
    anchors[:, 2] = anchors[:, 3] * ratios

    anchors[:, 0::2] -= (anchors[:, 2] * 0.5).repeat(2).view(2, -1).T
    anchors[:, 1::2] -= (anchors[:, 3] * 0.5).repeat(2).view(2, -1).T
    return anchors


def _project_anchors(fmap_shape: Sequence[int], stride: int, anchors: torch.Tensor) -> torch.Tensor:
    """
    project the calculated anchors in each (W x H) positions of a feature map.
    Args:
        fmap_shape (Tuple[int, int]): shape of the feature map (W x H)
        stride (int): stride of the feature map (downscale ratio to original image size)
        anchors (torch.Tensor): calculated anchors for a given stride

    Returns:
        torch.Tensor: anchor over all locations of a feature map. Given A anchors, the shape
        would be `(A x H x W, 4)`
    """
    fw, fh = fmap_shape[:2]
    x_mids = (torch.arange(fw) + 0.5) * stride
    y_mids = (torch.arange(fh) + 0.5) * stride
    x_mids = x_mids.repeat(fh, 1).T.flatten()
    y_mids = y_mids.repeat(fw)
    # xy = torch.stack([x_mids, y_mids]).T
    xy = torch.stack([y_mids, x_mids]).T
    xyxy = torch.cat([xy, xy], dim=1)
    # n_pos = xyxy.size(0)

    xyxy = xyxy.repeat_interleave(anchors.size(0), dim=0)
    anchors = anchors.repeat(fw * fh, 1)

    grid_anchors = xyxy + anchors
    return grid_anchors


if __name__ == "__main__":
    anchor = MultiBoxPrior()
    img = torch.randn(4, 3, 256, 256)
    anchors = anchor(img)
    print(anchors.size())
    print(anchors.sort(dim=-1))
