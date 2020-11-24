"""
__author__: bishwarup307
Created: 21/11/20
"""

import os
from pathlib import Path
from typing import Any, Union, Optional, Tuple

import torch
import torchvision


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
        t.size(-1) == 4
    ), "input tensor must be of size `(N, 4)` with format `(xmin, ymin, xmax, ymax)`"
    mins = t[..., :2]
    maxs = t[..., 2:]
    return torch.cat([(mins + maxs) / 2, maxs - mins], dim=-1)


def ccwh_to_xyxy(t: torch.Tensor) -> torch.Tensor:
    """
    converts bbox coordinates from `(x_center, y_center, width, height)` to `(xmin, ymin, xmax, ymax)`
    """
    assert (
        t.size(-1) == 4
    ), "input tensor must be of size `(N, 4)` with format `(x_center, y_center, width, height)`"
    cc = t[..., :2]
    wh = t[..., 2:]
    return torch.cat([cc - wh / 2, cc + wh / 2], dim=-1)


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
        boxes1 (torch.Tensor): box1 of the shape `(N, 4)`
        boxes2 (torch.Tensor): box1 of the shape `(M, 4)`
        eps (float): a constant to avoid zero division

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

    union = boxes1_area[:, None] + boxes2_area - intersection
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
    """
    Calculates target classes and coordinates (offsets to anchors) given a pair of anchor boxes and
    ground truth boxes along with their class labels. Any anchor boxes with IoU (with gt boxes)
    over `neg_threshold` and below `pos_threshold` are ignored for loss calculation and assigned a
    label as specified by `ignore_index`.
    Args:
        anchors (torch.Tensor): anchor boxes. shape `(N, 4)` where N = total number of anchors
        over each pixel of the feature map
        gt_boxes (torch.Tensor): ground truth boxes. shape `(T, 4)` where T is the number of
        ground truth instances.
        gt_cls (torch.Tensor): ground truth classes corresponding to each bounding box. shape `(T, )`
        pos_threshold (float): IoU threshold for positive class assignment
        neg_threshold (float): IoU threshold for negative class (background) assignment
        ignore_index (int): class index for the ignored anchor boxes

    Returns:

    """
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


def batched_nms(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    conf_threshold: Optional[float] = 0.05,
    nms_threshold: Optional[float] = 0.4,
):
    """
    Performs non-max suppression (NMS) on each batch level. This is done using torchvision`batched_nms` method.
    However, torchvision `batched_nms` performs NMS over a certain category index and here we need to take
    care of two different things, namely the image index and the class index to derive a unique category index to
    avoid mixing boxes across different images and/or different classes.

    The way the category index is derived is
    pretty straightforward (somewhat hacky). For a batch of N images with C number of classes we first filter the
    predicted boxes for their confidence scores and discard all the boxes below a certain threshold as specified
    by `conf_threshold`. Then we calculate the number of instances per image and initialize a vector which
    simply represents the image index for each of the instances. In order to derive the category index, we finally
    multiply this vector with the corresponding class indices. In order to make sure each (image_id, class_id) tuple
    is represented uniquely, we increment the class index to start from one (as a 0 class index will always result
    in 0 category index and that is not intended).


    Args:
        logits (torch.Tensor): logits from the model. Shape `(N, A, C)` where N, A and C are the batch size,
        number of anchors and number of classes respectively.
        boxes (torch.Tensor): box coordinates corresponding to the clases. Shape `(N, A, 4)`.
        conf_threshold (Optional[float]): Confidence threshold for the predictions. Defaults to 0.05.
        nms_threshold (Optional[float]): IoU threshold for NMS. Defaults to 0.4.

    Returns:

    """

    num_classes = logits.size(-1)

    scores, class_indices = logits.max(dim=2)
    mask = scores > conf_threshold
    instances_per_image = mask.sum(dim=1)

    selected_class_indices = class_indices[mask]
    image_ids = torch.arange(num_classes, num_classes + logits.size(0))
    image_ids = torch.repeat_interleave(image_ids, instances_per_image)
    category_idx = image_ids * (selected_class_indices + 1)
    selected_bboxes = boxes[mask]

    keep_indices = torchvision.ops.boxes.batched_nms(
        selected_bboxes, scores[mask], category_idx, nms_threshold
    )
    nms_image_idx = image_ids[keep_indices] - num_classes
    nms_bboxes = selected_bboxes[keep_indices]
    nms_scores = scores[mask][keep_indices]
    nms_classes = selected_class_indices[keep_indices]
    return nms_image_idx, nms_bboxes, nms_classes, nms_scores


if __name__ == "__main__":
    print(ifnone(None))
