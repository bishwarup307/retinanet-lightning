"""
__author__: bishwarup307
Created: 26/11/20
"""
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import albumentations as A
from omegaconf import ListConfig, DictConfig
import torch

from retinanet import utils


class Resizer:
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, size: Tuple[int, int], resize_mode: str = "letterbox"):
        if resize_mode not in ("letterbox", "minmax"):
            raise ValueError(
                f"`resize_mode` should be either `letterbox` or `minmax`, got {resize_mode}"
            )
        self.size = size
        self.resize_fn = {"letterbox": letterbox, "minmax": min_max_resize}[resize_mode]

    def __call__(self, sample) -> Dict:
        resize_params = dict()

        image = sample["img"]
        rsz_img, scale, offset_x, offset_y = self.resize_fn(image, self.size)
        resize_params["img"] = rsz_img
        resize_params["scale"] = scale
        resize_params["offset_x"] = offset_x
        resize_params["offset_y"] = offset_y
        try:
            annots = sample["annot"]
            annots[:, :4] *= scale
            annots[:, 0] += offset_x
            annots[:, 1] += offset_y
            annots[:, 2] += offset_x
            annots[:, 3] += offset_y
            resize_params["annot"] = annots
        except KeyError:
            pass

        return resize_params


class Normalizer:
    def __init__(self, mean: Optional[List[float]], std: Optional[List[float]]):
        self.mean = mean
        self.std = std

    def __call__(self, img: np.ndarray) -> np.ndarray:
        normalized_img = img.astype(np.float32) / 255.0
        if self.mean is not None and self.std is not None:
            normalized_img = (normalized_img - self.mean) / self.std
        return normalized_img


class UnNormalizer:
    def __init__(self, mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
        self.mean = utils.ifnone(mean, [0.485, 0.456, 0.406])
        self.std = utils.ifnone(std, [0.229, 0.224, 0.225])

    def __call__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Augmenter:
    def __init__(self, transforms: Dict, bbox_mode: str = "pascal_voc"):
        self.transforms = dict()
        for k, v in transforms.items():
            # if isinstance(v, DictConfig):
            #     for k_, v_ in v.items():
            #         if isinstance(v_, ListConfig):
            #             v[k_] = tuple(v_)
            #             a[k_] = tuple(v_)
            #     print(v)
            #     print(a)
            self.transforms[k] = self._parse_config(v)
        self.bbox_mode = bbox_mode
        self._compose_augments()

    @staticmethod
    def _parse_config(aug):
        if isinstance(aug, DictConfig):
            a = dict()
            for k, v in aug.items():
                if isinstance(v, ListConfig):
                    a[k] = tuple(v)
                else:
                    a[k] = v
            return a
        return aug

    def _compose_augments(self):
        augs = []
        for name, params in self.transforms.items():
            if name == "horizontal_flip":
                augs.append(A.HorizontalFlip(**params))
            if name == "vertical_flip":
                augs.append(A.VerticalFlip(**params))
            if name == "shift_scale_rotate":
                augs.append(A.ShiftScaleRotate(**params))
            if name == "gamma":
                augs.append(A.RandomGamma(**params))
            if name == "sharpness":
                augs.append(A.IAASharpen(**params))
            if name == "gaussian_blur":
                augs.append(A.GaussianBlur(**params))
            if name == "super_pixels":
                augs.append(A.IAASuperpixels(**params))
            if name == "additive_noise":
                augs.append(A.IAAAdditiveGaussianNoise(**params))
            if name == "perspective":
                augs.append(A.IAAPerspective(**params))
            if name == "color_jitter":
                augs.append(A.ColorJitter(**params))
            if name == "brightness":
                augs.append(A.RandomBrightness(**params))
            if name == "contrast":
                augs.append(A.RandomContrast(**params))

            if name == "rgb_shift":
                augs.append(A.RGBShift(**params))
            if name == "cutout":
                augs.append(A.Cutout(**params))
        self.trsf = A.Compose(
            augs,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["category_ids"],
                min_visibility=self.transforms["min_visibility"],
                min_area=self.transforms["min_area"],
            ),
        )

    def __call__(self, sample) -> Dict:

        # if len(self.transforms) <= 2:
        #     return sample

        transformed = self.trsf(
            image=sample["img"],
            bboxes=sample["annot"][:, :-1],
            category_ids=sample["annot"][:, -1],
        )
        if len(transformed["bboxes"]):
            annot = np.concatenate(
                [
                    np.array(transformed["bboxes"]),
                    np.array(transformed["category_ids"])[..., np.newaxis],
                ],
                axis=1,
            )
        else:
            annot = np.zeros((0, 5))

        sample["img"] = transformed["image"]
        sample["annot"] = annot

        return sample


def letterbox(image: np.ndarray, expected_size: Tuple[int, int], fill_value: int = 0):
    """
    Choose the smaller scale between width and height maintaining the AR. pad the other side to meet
    the expected size.
    Args:
        image (np.ndarray): image
        expected_size (Tuple[int, int]): expected image size (height, width)
        fill_value (int): padding fill

    Returns: Tuple[np.ndarray, float, float, float]. in the order (resized_image, scale, offset_x, offset_y)
    """
    ih, iw, _ = image.shape
    eh, ew = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    new_img = np.full((eh, ew, 3), fill_value, dtype=np.uint8)

    offset_x, offset_y = (ew - nw) // 2, (eh - nh) // 2

    new_img[offset_y : offset_y + nh, offset_x : offset_x + nw, :] = image.copy()
    return new_img, scale, offset_x, offset_y


def min_max_resize(image: np.ndarray, sizes: Tuple[int, int]):
    """
    The images are rescaled (by default) such that the smallest side equals min(sizes). If the largest side is
    then still larger than max(sizes) pixels, it is scaled down further such that the largest side is equal
    to max(sizes) pixels.
    Args:
        image (np.ndarray):
        sizes (Tuple[int, int]): (min_size, max_size)

    Returns: Tuple[np.ndarray, float, float, float]. in the order (resized_image, scale, offset_x, offset_y)
                the offsets are zero in this case but included in the fn to have cosistent API with
                letterbox.
    """
    offset_x, offset_y = 0, 0
    min_size, max_size = sizes
    rows, cols, cns = image.shape
    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_size / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)

    if largest_side * scale > max_size:
        scale = max_size / largest_side
    # resize the image with the computed scale
    image = cv2.resize(
        image, (int(round((cols * scale))), int(round(rows * scale))), interpolation=cv2.INTER_CUBIC
    )
    rows, cols, cns = image.shape

    pad_w = (32 - rows % 32) % 32
    pad_h = (32 - cols % 32) % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.uint8)
    new_image[:rows, :cols, :] = image
    return new_image, scale, offset_x, offset_y
