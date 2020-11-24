"""
__author__: bishwarup307
Created: 21/11/20
"""

import os
from pathlib import Path
from typing import Any, Union, Optional, Tuple

import torch


def ifnone(x: Any, y: Any):
    val = x if x is not None else y
    return val


def isfile(path: Union[str, os.PathLike]):
    return Path(path).is_file()


def ifdir(path: Union[str, os.PathLike], alt_path: Union[str, os.PathLike]):
    return path if Path(path).is_dir() else alt_path


def to_tensor(x: Any):
    try:
        return torch.tensor(x)
    except ValueError as exc:
        print("invalid type for tensor conversion")
        raise exc


def xyxy_to_ccwh(t: torch.Tensor) -> torch.Tensor:
    """
    converts bbox coordinates from `(xmin, ymin, xmax, ymax)` to `(x_center, y_center, width, height)`
    """
    assert (
        t.size(1) == 4
    ), "input tensor must be of size `(N, 4)` with format `(xmin, ymin, xmax, ymax)`"
    mins = t[..., :2]
    maxs = t[..., 2:]
    return torch.cat([(mins + maxs) / 2, maxs - mins], 1)


def ccwh_to_xyxy(t: torch.Tensor) -> torch.Tensor:
    """
    converts bbox coordinates from `(x_center, y_center, width, height)` to `(xmin, ymin, xmax, ymax)`
    """
    assert (
        t.size(1) == 4
    ), "input tensor must be of size `(N, 4)` with format `(x_center, y_center, width, height)`"
    cc = t[..., :2]
    wh = t[..., 2:]
    return torch.cat([cc - wh / 2, cc + wh / 2], 1)


def _is_xyxy(boxes: torch.Tensor) -> bool:
    is_valid_width = (boxes[:, 2] - boxes[:, 0] > 0).all()
    is_valid_height = (boxes[:, 3] - boxes[:, 1] > 0).all()
    return is_valid_width and is_valid_height


def bbox_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor, eps: Optional[float] = 1e-6
) -> torch.Tensor:
    """
    Calculates Intersection over Union (IoU), also known as Jaccard Index, between a pair of bounding boxes not
    necessarily of the same length. If the `boxes1` shape is `(N, 4)` and `boxes2` shape is `(M, 4)`, the
    ious are of shape `(N, M)`, where
        iou[i, j] = IoU of i-th box of boxes1 with j-th box of boxes2.
    Args:
        boxes1 [torch.Tensor]: box1 of the shape `(N, 4)`
        boxes2 [torch.Tensor]: box1 of the shape `(M, 4)`
        eps: a constant to avoid zero division

    Returns:
        IoUs between the boxes
    """
    assert (
        boxes1.ndim == 2
    ), f"Invalid box dimension, requires of the shape `(N, 4)`, got {list(boxes1.shape)}"
    assert (
        boxes2.ndim == 2
    ), f"Invalid box dimension, requires of the shape `(N, 4)`, got {list(boxes2.shape)}"

    assert _is_xyxy(
        boxes1
    ), "expects bboxes to be in the format `(xmin, ymin, xmax, ymax)` at position 0"
    assert _is_xyxy(
        boxes2
    ), "expects bboxes to be in the format `(xmin, ymin, xmax, ymax)` at position 1"

    # intersection = min(max(coordinate)) - max(min(coordinate))
    iw = torch.min(boxes1[:, None, 2], boxes2[:, 2]) - torch.max(
        boxes1[:, None, 0], boxes2[:, 0]
    )
    ih = torch.min(boxes1[:, None, 3], boxes2[:, 3]) - torch.max(
        boxes1[:, None, 1], boxes2[:, 1]
    )
    # iw and ih are negative if there is no intersection between widths and heights
    iw = torch.clamp(iw, 0.0)
    ih = torch.clamp(ih, 0.0)
    intersection = iw * ih

    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = boxes1_area + boxes2_area[:, None] - intersection
    union = torch.clamp(union, eps)
    ious = intersection / union
    return ious


def _calculate_offsets(anchors: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    targets[:, :2] = (targets[:, :2] - anchors[:, :2]) / anchors[:, 2:]
    targets[:, 2:] = torch.log(targets[:, 2:] / anchors[:, 2:])
    return targets


def get_anchor_labels(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_cls: torch.Tensor,
    pos_threshold: Optional[float] = 0.5,
    neg_threshold: Optional[float] = 0.4,
    ignore_index: Optional[int] = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ious = bbox_iou(anchors, gt_boxes)
    max_iou, max_idx = ious.max(dim=1)

    gt_boxes = xyxy_to_ccwh(gt_boxes)
    anchors = xyxy_to_ccwh(anchors)

    target_boxes = gt_boxes[max_idx]
    target_boxes = _calculate_offsets(anchors, target_boxes)

    target_classes = 1 + gt_cls[max_idx]
    target_classes[max_iou <= neg_threshold] = 0.0
    null_indices = (max_iou > neg_threshold) & (max_iou < pos_threshold)
    target_classes[null_indices] = ignore_index

    return target_boxes, target_classes


if __name__ == "__main__":
    print(ifnone(None))
