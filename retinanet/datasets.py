"""
__author__: bishwarup307
Created: 23/11/20
"""
import glob
import os
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Union, Callable

import cv2
import numpy as np
import torch
from PIL import Image
from coco.coco import COCO
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from retinanet.anchors import MultiBoxPrior
from retinanet.augments import Resizer, Augmenter, Normalizer
from retinanet.utils import get_anchor_labels, isfile, ifnone

import pytorch_lightning as pl
from omegaconf import DictConfig


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super(DataModule, self).__init__()
        if cfg.Dataset.dataset != "coco":
            print(f"only COCO dataset is supported at present, got {cfg.Dataset.dataset}")
            raise NotImplementedError
        self.cfg = cfg
        self.image_dir = Path(cfg.Dataset.root) / "images"
        self.annotation_dir = Path(cfg.Dataset.root) / "annotations"
        self.image_size = cfg.Dataset.image_size[:2]

        self.train_name = ifnone(cfg.Dataset.train_name, "train")
        self.val_name = ifnone(cfg.Dataset.val_name, "val")
        self.test_name = ifnone(cfg.Dataset.test_name, "test")
        self.label_ext = "json" if cfg.Dataset.dataset == "coco" else None
        self._register_paths()
        if self.train_label_path is None:
            raise IOError(f"Could not load {self.train_label_path}, no such file on disk.")

    def _register_paths(self):
        if self.image_dir.joinpath(self.train_name).is_dir():
            self.train_image_dir = self.image_dir.joinpath(self.train_name)
            if self.image_dir.joinpath(self.val_name).is_dir():
                self.val_image_dir = self.image_dir.joinpath(self.val_name)
            if self.image_dir.joinpath(self.test_name).is_dir():
                self.test_image_dir = self.image_dir.joinpath(self.test_name)
        else:
            self.train_image_dir = self.image_dir
            self.val_image_dir = self.image_dir
            self.test_image_dir = self.image_dir

        self.train_label_path = isfile(self.annotation_dir.joinpath(f"{self.train_name}.{self.label_ext}"))
        self.val_label_path = isfile(self.annotation_dir.joinpath(f"{self.val_name}.{self.label_ext}"))
        self.test_label_path = isfile(self.annotation_dir.joinpath(f"{self.test_name}.{self.label_ext}"))

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = CocoDataset(
                image_dir=self.train_image_dir,
                json_path=self.train_label_path,
                image_size=self.image_size,
                nsr=self.cfg.Dataset.nsr,
            )
            self.num_classes = len(self.train_dataset.coco.getCatIds())
            self.anchors = self.train_dataset.anchors
            self.train_samples = len(self.train_dataset)

            if self.val_label_path is not None:
                self.val_dataset = CocoDataset(
                    image_dir=self.val_image_dir,
                    json_path=self.val_label_path,
                    image_size=self.image_size,
                    nsr=self.cfg.Dataset.nsr,
                    train=False,
                )
            else:
                self.val_dataset = None

        if stage == "test" or stage is None:
            if self.test_label_path is not None:
                self.test_dataset = CocoDataset(
                    image_dir=self.test_image_dir,
                    json_path=self.test_label_path,
                    image_size=self.image_size,
                    train=False,
                )
            else:
                self.test_dataset = None

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        if self.cfg.Dataset.nsr is not None:
            sampler = WeightedRandomSampler(self.train_dataset.weights, len(self.train_dataset), replacement=True)
            return DataLoader(
                self.train_dataset,
                sampler=sampler,
                batch_size=self.cfg.Trainer.batch_size.train,
                num_workers=self.cfg.Trainer.workers,
                pin_memory=self.cfg.Trainer.gpus > 0,
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.Trainer.batch_size.train,
            num_workers=self.cfg.Trainer.workers,
            pin_memory=self.cfg.Trainer.gpus > 0,
            shuffle=True,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        if self.val_dataset is not None:
            return DataLoader(
                self.val_dataset,
                batch_size=self.cfg.Trainer.batch_size.val,
                num_workers=self.cfg.Trainer.workers,
                pin_memory=self.cfg.Trainer.gpus > 0,
                shuffle=False,
            )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.cfg.Trainer.batch_size.test,
                num_workers=self.cfg.Trainer.workers,
                pin_memory=self.cfg.Trainer.gpus > 0,
                shuffle=False,
            )


class CocoDataset(Dataset):
    """
    Creates a `torch.utils.Dataset` object from COCO annotations.

    Args:
        image_dir (str): Path to the images
        json_path (str): Path to the coco json file
        image_size Tuple[int, int]: image width and height as required by the model. Required for resizing images
        normalize (Optional[Dict[str, List[float]): normalization factors for the image channels
        transform (Optional[List[Callable]]): a list of transforms
        train (bool): whether the dataset is used for training or validation/inference
        nsr (float): negative sampling rate at batch level

    """

    def __init__(
        self,
        image_dir: str,
        json_path: str,
        image_size: Tuple[int, int],
        normalize: Optional[Dict] = None,
        transform: Optional[List[Callable]] = None,
        train: bool = True,
        nsr: float = None,
    ):

        self.image_dir = image_dir
        self.transform = transform
        self.image_size = image_size
        self.train = train

        try:
            self.normalize_mean = normalize["mean"]
            self.normalize_std = normalize["std"]
        except TypeError:
            self.normalize_mean, self.normalize_std = (
                [0.485, 0.456, 0.406],
                [
                    0.229,
                    0.224,
                    0.225,
                ],
            )

        self.coco = COCO(json_path)
        self.image_ids = self.coco.getImgIds()
        self.return_ids = not train
        self.nsr = ifnone(nsr, 1.0)

        self.classes = dict()
        self.coco_label_map = dict()
        self.labels = dict()
        self.coco_labels = dict()
        self.coco_labels_inverse = dict()
        self._load_classes()
        m = MultiBoxPrior()
        sample_input = torch.randn(1, 3, *image_size)
        self.anchors = m(sample_input)

        self._obtain_weights()
        # print(f"number of classes: {self.num_classes}")

    def __len__(self):
        return len(self.image_ids)

    def _clip_annotations(self, sample):
        # albumentation complains if bbox coordinates are equal to image shape in
        # either dimensions
        # https://github.com/albumentations-team/albumentations/issues/459
        h, w = sample["img"].shape[:2]
        annots = sample["annot"]
        annots[:, 0:4:2] = annots[:, 0:4:2].clip(1, w)
        annots[:, 1:4:2] = annots[:, 1:4:2].clip(1, h)
        sample["annot"] = annots
        return sample

    def __getitem__(self, idx):
        img = self._load_image(idx, normalize=False)  # load image
        annot = self._load_annotations(idx)
        sample = {"img": img, "annot": annot}
        if self.image_size is not None:
            resize = Resizer(self.image_size)  # resize
            sample = resize(sample)

        sample = self._clip_annotations(sample)

        if self.transform is not None:
            augment = Augmenter(self.transform)
            sample = augment(sample)

        if self.return_ids:
            # return self._to_tensor(sample), self.image_ids[idx]
            sample["image_id"] = self.image_ids[idx]

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
            self.coco_label_map[c["id"]] = c["name"]
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
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
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

        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def _to_tensor(self, sample, normalize=True):
        if normalize:
            normalizer = Normalizer(self.normalize_mean, self.normalize_std)
            sample["img"] = normalizer(sample["img"])

        sample["img"] = torch.from_numpy(sample["img"].astype(np.float32))
        sample["annot"] = torch.from_numpy(sample["annot"].astype(np.float32))
        # if self.train:
        gt_boxes, gt_cls = get_anchor_labels(self.anchors, sample["annot"][:, :4], sample["annot"][:, 4])
        if self.train:
            return sample["img"].permute(2, 0, 1).contiguous(), gt_boxes, gt_cls
        return (
            sample["img"].permute(2, 0, 1).contiguous(),
            gt_boxes,
            gt_cls,
            sample["scale"],
            sample["offset_x"],
            sample["offset_y"],
            sample["image_id"],
        )

    def _coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]


class TestDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: Tuple[int, int],
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None,
    ):
        exts = ["jpg", "png", "tiff", "JPG", "jpeg"]
        self.images = [x for x in glob.glob(os.path.join(root, "*")) for ext in exts if x.endswith(ext)]

        if not len(self.images):
            raise ValueError(f"The specified diretory {root} does not contain any image files")

        self.image_ids = torch.arange(len(self.images))
        self.image_size = image_size

        self.mean = ifnone(normalize_mean, [0.485, 0.456, 0.406])
        self.std = ifnone(normalize_std, [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        if idx > len(self):
            idx = np.random.choice(range(len(self)))
        img = np.array(Image.open(self.images[idx]))
        sample = {"img": img}
        resize = Resizer(self.image_size)  # resize
        sample = resize(sample)

        normalizer = Normalizer(self.mean, self.std)
        sample["img"] = normalizer(sample["img"])
        sample["img"] = torch.from_numpy(sample["img"].astype(np.float32))

        sample["image_id"] = self.image_ids[idx]
        return (
            sample["img"].permute(2, 0, 1).contiguous(),
            sample["scale"],
            sample["offset_x"],
            sample["offset_y"],
            sample["image_id"],
        )


def list_collate(batch: List):
    image_ids, images, labels, scales, offset_x, offset_y = [], [], [], [], [], []

    for instance in batch:
        images.append(instance["img"])
        labels.append(instance["annot"])
        scales.append(instance["scale"])
        offset_x.append(instance["offset_x"])
        offset_y.append(instance["offset_y"])
        image_ids.append(instance["image_id"])
    return (
        torch.stack(images).permute(0, 3, 1, 2).contiguous(),
        labels,
        scales,
        offset_x,
        offset_y,
        image_ids,
    )


def collater(batch: List):
    imgs = [b[0] for b in batch]
    boxes = [b[1] for b in batch]
    classes = [b[2] for b in batch]

    heights = [int(s.shape[1]) for s in imgs]
    widths = [int(s.shape[2]) for s in imgs]
    batch_size = len(imgs)

    max_width, max_height = max(widths), max(heights)
    batch_imgs = torch.zeros(batch_size, 3, max_height, max_width)

    for i in range(batch_size):
        img = imgs[i]
        batch_imgs[i, :, : int(img.shape[1]), : int(img.shape[2])] = img

    batch_boxes = torch.stack(boxes)
    batch_classes = torch.stack(classes)
    return batch_imgs, batch_boxes, batch_classes
