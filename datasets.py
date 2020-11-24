"""
__author__: bishwarup307
Created: 23/11/20
"""
import os
from pathlib import Path
from typing import Union, Optional, Dict, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from utils import ifdir


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(
        self,
        image_dir: str,
        json_path: str,
        image_size: Sequence[int],
        normalize: Optional[Dict] = None,
        transform: Optional[Dict] = None,
        return_ids: bool = False,
        nsr: float = None,
    ):
        """
        Args:
            image_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.root_dir = root_dir
        # self.set_name = set_name
        self.image_dir = image_dir
        self.transform = transform
        self.image_size = image_size
        try:
            self.normalize_mean = normalize["mean"]
            self.normalize_std = normalize["std"]
        except TypeError:
            self.normalize_mean, self.normalize_std = None, None

        self.coco = COCO(json_path)
        self.image_ids = self.coco.getImgIds()
        self.return_ids = return_ids
        self.nsr = nsr if nsr is not None else 1.0

        self.classes = {}
        self.labels = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        self._load_classes()

        self._obtain_weights()
        # print(f"number of classes: {self.num_classes}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self._load_image(idx, normalize=False)  # load image
        annot = self._load_annotations(idx)
        sample = {"img": img, "annot": annot}
        if self.image_size is not None:
            resize = Resizer(self.image_size)  # resize
            sample = resize(sample)

        if self.transform is not None:
            sample = _transform_image(sample, self.transform)

        if self.return_ids:
            return self._to_tensor(sample), self.image_ids[idx]

        return self._to_tensor(sample)

    def _obtain_weights(self):
        weights = []
        for imid in self.image_ids:
            anns = self.coco.getAnnIds([imid])
            if anns:
                weights.append(1)
            else:
                weights.append(self.nsr)
        self.weights = weights

    def _load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])

        for c in categories:
            self.coco_labels[len(self.classes)] = c["id"]
            self.coco_labels_inverse[c["id"]] = len(self.classes)
            self.classes[c["name"]] = len(self.classes)

        # also load the reverse (label -> name)
        for key, value in self.classes.items():
            self.labels[value] = key

    def _load_image(self, image_index, normalize=True):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.image_dir, image_info["file_name"])
        img = np.array(Image.open(path))

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if not normalize:
            return img

        return img.astype(np.float32) / 255.0

    def _load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_index], iscrowd=False
        )
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a["bbox"][2] < 1 or a["bbox"][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a["bbox"]
            annotation[0, 4] = self._coco_label_to_label(a["category_id"])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def _to_tensor(self, sample, normalize=True):
        if normalize:
            normalizer = Normalizer(self.normalize_mean, self.normalize_std)
            sample["img"] = normalizer(sample["img"])

        sample["img"] = torch.from_numpy(sample["img"].astype(np.float32))
        sample["annot"] = torch.from_numpy(sample["annot"].astype(np.float32))
        return sample

    def _coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]
