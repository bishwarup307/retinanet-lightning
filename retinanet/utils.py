"""
__author__: bishwarup307
Created: 21/11/20
"""
import copy
import importlib
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Any, Union, Optional, Tuple

import colorama
import cv2
import numpy as np
import torch
import torchvision
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import torch
from torch.utils.data import Dataset
import torch.distributed as dist

from retinanet import augments

LOG_COLORS = {
    logging.ERROR: colorama.Fore.RED,
    logging.WARNING: colorama.Fore.YELLOW,
    logging.INFO: colorama.Fore.GREEN,
    logging.DEBUG: colorama.Fore.WHITE,
}


def ifnone(x: Any, y: Any):
    """
    returns x if x is none else returns y
    """
    val = x if x is not None else y
    return val


def isfile(path: Union[str, os.PathLike]):
    """
    returns tha path provided as argument if the file exists. Returns `None` if not.
    """
    if Path(path).is_file():
        return path
    return None


def ifdir(path: Union[str, os.PathLike], alt_path: Union[str, os.PathLike]):
    return path if Path(path).is_dir() else alt_path


def to_tensor(x: Any):
    try:
        return torch.tensor(x)
    except ValueError as exc:
        print("invalid type for tensor conversion")
        raise exc


# https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.
    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path.
    Returns:
        Extracted object.
    Raises:
        AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError("Object `{}` cannot be loaded from `{}`.".format(obj_name, obj_path))
    return getattr(module_obj, obj_name)


def get_device_config(gpus: int, tpus: int):
    if tpus > 0:
        return None, tpus
    if gpus > 0:
        gpus = min(gpus, torch.cuda.device_count())
        return gpus, None
    return None, None


def get_total_steps(n_samples: int, batch_size: int) -> int:
    div, mod = divmod(n_samples, batch_size)
    if mod == 0:
        return div
    return div + 1


class ColorFormatter(logging.Formatter):
    def format(self, record, *args, **kwargs):
        # if the corresponding logger has children, they may receive modified
        # record, so we want to keep it intact
        new_record = copy.copy(record)
        if new_record.levelno in LOG_COLORS:
            # we want levelname to be in different color, so let's modify it
            new_record.levelname = "{color_begin}{level}{color_end}".format(
                level=new_record.levelname,
                filename=new_record.filename,
                color_begin=LOG_COLORS[new_record.levelno],
                color_end=colorama.Style.RESET_ALL,
            )
        # now we can let standart formatting take care of the rest
        return super(ColorFormatter, self).format(new_record, *args, **kwargs)


def get_logger(
    name,
    filepath: Optional[Union[str, os.PathLike]] = None,
    level: Optional[str] = "debug",
):
    log_level = {"info": logging.INFO, "debug": logging.DEBUG, "error": logging.ERROR}

    logger = logging.getLogger(name)
    logger.setLevel(log_level.get(level.lower(), logging.INFO))
    if filepath is not None:
        file_handler = logging.FileHandler(filepath)
        file_handler.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = ColorFormatter("[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] -> %(message)s")
    if filepath is not None:
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def get_callbacks(callback_config: DictConfig):
    callbacks = []
    for callback, params in callback_config.items():
        if not params.enabled:
            continue
        if callback == "checkpoint":
            callbacks.append(
                ModelCheckpoint(
                    save_top_k=params.save_top_k,
                    monitor=params.monitor,
                    mode=params.mode,
                    verbose=params.verbose,
                )
            )
        if callback == "early_stopping":
            callbacks.append(
                EarlyStopping(
                    patience=params.patience,
                    monitor=params.monitor,
                    mode=params.mode,
                    verbose=params.verbose,
                )
            )
        if callback == "lr_monitor":
            callbacks.append(LearningRateMonitor(logging_interval=params.logging_interval))
    return callbacks


def coco_to_preds(coco: Any):
    results = dict()
    coco_dict = coco.dataset
    for im in coco_dict["images"]:
        im_id = im["id"]
        filename = im["file_name"]
        anns = coco.loadAnns(coco.getAnnIds(im_id))
        for instance in anns:
            bbox = list(map(int, instance["bbox"]))
            score = instance["score"]
            class_index = instance["category_id"]
            record = {"bbox": bbox, "confidence": score, "class_index": class_index}
            if filename in results:
                results[filename].append(record)
            else:
                results[filename] = [record]
    return results


# https://github.com/pytorch/vision/blob/e337103f2c222528f505772f1289dfee06117de4/references/detection/utils.py#L75
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def xyxy_to_ccwh(t: torch.Tensor) -> torch.Tensor:
    """
    converts bbox coordinates from `(xmin, ymin, xmax, ymax)` to `(x_center, y_center, width, height)`
    """
    assert t.size(-1) == 4, "input tensor must be of size `(N, 4)` with format `(xmin, ymin, xmax, ymax)`"
    mins = t[..., :2]
    maxs = t[..., 2:]
    return torch.cat([(mins + maxs) / 2, maxs - mins], dim=-1)


def ccwh_to_xyxy(t: torch.Tensor) -> torch.Tensor:
    """
    converts bbox coordinates from `(x_center, y_center, width, height)` to `(xmin, ymin, xmax, ymax)`
    """
    assert t.size(-1) == 4, "input tensor must be of size `(N, 4)` with format `(x_center, y_center, width, height)`"
    cc = t[..., :2]
    wh = t[..., 2:]
    return torch.cat([cc - wh / 2, cc + wh / 2], dim=-1)


def _is_xyxy(boxes: torch.Tensor) -> bool:
    is_valid_width = (boxes[:, 2] - boxes[:, 0] > 0).all()
    is_valid_height = (boxes[:, 3] - boxes[:, 1] > 0).all()
    return is_valid_width and is_valid_height


def bbox_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: Optional[float] = 1e-6) -> torch.Tensor:
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
    assert boxes1.ndim == 2, f"Invalid box dimension, requires of the shape `(N, 4)`, got {list(boxes1.shape)}"
    assert boxes2.ndim == 2, f"Invalid box dimension, requires of the shape `(N, 4)`, got {list(boxes2.shape)}"

    assert _is_xyxy(boxes1), "expects bboxes to be in the format `(xmin, ymin, xmax, ymax)` at position 0"
    assert _is_xyxy(boxes2), f"expects bboxes to be in the format `(xmin, ymin, xmax, ymax)` at position 1 \n {boxes2}"

    # intersection = min(max(coordinate)) - max(min(coordinate))
    iw = torch.min(boxes1[:, None, 2], boxes2[:, 2]) - torch.max(boxes1[:, None, 0], boxes2[:, 0])
    ih = torch.min(boxes1[:, None, 3], boxes2[:, 3]) - torch.max(boxes1[:, None, 1], boxes2[:, 1])
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
    """Calculate offsets for bounding box regression. Offsets are calculated following the
    equations below:
    ..math:

            x'' = \frac{x'}{\sigma_{x}^{2}} = \frac{x_{gt} - x_{anchor}}{w_{anchor}} / \sigma_{X'}

            y'' = \frac{y'}{\sigma_{y}^{2}} = \frac{y_{gt} - y_{anchor}}{h_{anchor}} / \sigma_{Y'}

            w'' = \frac{w'}{\sigma_{w}^{2}} = \ln \left[ \frac{w_{gt}}{w_{anchor}} \right] / \sigma_{W'}

            h'' = \frac{h'}{\sigma_{h}^{2}} = \ln \left[ \frac{h_{gt}}{h_{anchor}} \right] / \sigma_{H'}
    Args:
        anchors (torch.Tensor): anchor boxes (priors). Shape `(A, 4)`. the boxes must be in
        :math:`(x_{center}, y_{center}, width, height)` format.
        targets (torch.Tensor): ground truth boxes. Shape `(A, 4)`. the boxes must be in
        :math:`(x_{center}, y_{center}, width, height)` format.

    Returns:
        (torch.Tensor): the calculated offsets (deltas) for :math:`(x_{center}, y_{center}, width, height)`.
        Shape `(A, 4)`

    """
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
    if len(gt_boxes) == 0:
        target_boxes = torch.zeros_like(anchors)
        target_classes = torch.zeros(len(anchors))  # all anchors are assigned to background

    else:
        ious = bbox_iou(anchors, gt_boxes)
        max_iou, max_idx = ious.max(dim=1)

        gt_boxes = xyxy_to_ccwh(gt_boxes)
        anchors = xyxy_to_ccwh(anchors)

        target_boxes = gt_boxes[max_idx]
        target_boxes = _calculate_offsets(anchors, target_boxes)
        target_boxes = target_boxes / torch.tensor([0.1, 0.1, 0.2, 0.2]).float()  # scale with prior variance

        target_classes = 1 + gt_cls[max_idx]
        target_classes[max_iou <= neg_threshold] = 0.0
        null_indices = (max_iou > neg_threshold) & (max_iou < pos_threshold)
        target_classes[null_indices] = ignore_index

    return target_boxes, target_classes


def batched_nms(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    conf_threshold: Optional[float] = 0.1,
    nms_threshold: Optional[float] = 0.5,
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
        nms_image_idx (torch.Tensor): the location index of the image in the given batch. Shape `(N,)`
        nms_boxes (torch.Tensor): the boxes after NMS. Shape `(P, 4)` where P is the number of boxes after NMS
        nms_classes (torch.Tensor) : the class indices for the `nms_boxes`. Shape `(N,)`
        nms_scores (torch.Tensor): the confidence score for the `nms_boxes`. Shape `(N, )`
    """

    num_classes = logits.size(-1)
    # print(f"batched_nms >> num_classes: {num_classes}")
    # print(logits.size())
    scores, class_indices = logits.max(dim=2)
    # print(f"batched_nms >> max_score: {scores.max()}")
    # print(f"batched_nms >> min_score: {scores.min()}")
    mask = scores > conf_threshold
    instances_per_image = mask.sum(dim=1)

    selected_class_indices = class_indices[mask]
    image_ids = torch.arange(num_classes, num_classes + logits.size(0)).type_as(class_indices)
    image_ids = torch.repeat_interleave(image_ids, instances_per_image)
    category_idx = image_ids * (selected_class_indices + 1)
    selected_bboxes = boxes[mask]

    # print(selected_bboxes.size())
    # print(category_idx.size())
    # print(scores[mask].size())

    keep_indices = torchvision.ops.boxes.batched_nms(selected_bboxes, scores[mask], category_idx, nms_threshold)
    nms_image_idx = image_ids[keep_indices] - num_classes
    nms_bboxes = selected_bboxes[keep_indices]
    nms_scores = scores[mask][keep_indices]
    nms_classes = selected_class_indices[keep_indices]
    return nms_image_idx, nms_bboxes, nms_classes, nms_scores


def offset_to_bbox(offsets: torch.Tensor, anchors: torch.Tensor):
    cc = offsets[..., :2]
    wh = offsets[..., 2:]

    cc = cc * anchors[..., 2:] + anchors[..., :2]
    wh = wh.exp() * anchors[..., 2:]
    boxes_ccwh = torch.cat([cc, wh], dim=2)
    boxes_xyxy = ccwh_to_xyxy(boxes_ccwh)
    return boxes_xyxy


def visualize_random_sample(dataset: Dataset, train: bool = True, unnormalize: bool = True):
    n = len(dataset)
    index = random.choice(np.arange(n))
    if train:
        img, gt_boxes, gt_cls = dataset[index]
    else:
        img, gt_boxes, gt_cls, *_ = dataset[index]
    if unnormalize:
        mean, std = dataset.normalize_mean, dataset.normalize_std
        unm = augments.UnNormalizer(mean, std)
        img = unm(img)
    img = img.numpy().transpose(1, 2, 0)
    img = (img * 255.0).astype(np.uint8)
    img = np.ascontiguousarray(img)
    assigned_indices = gt_cls > 0
    labels = gt_cls[assigned_indices]
    assigned_anchors = dataset.anchors[assigned_indices]
    # assigned_anchors = xyxy_to_ccwh(assigned_anchors)
    # print(labels)
    for i, lbl in enumerate(labels):
        coords = list(map(int, assigned_anchors[i].numpy().tolist()))
        # print(coords)
        # bb.add(img, *coords, "a", "blue")
        cv2.rectangle(
            img,
            (coords[0], coords[1]),
            (coords[2], coords[3]),
            (0, 255, 0),
            thickness=1,
        )

    return img


if __name__ == "__main__":
    print(ifnone(None))
