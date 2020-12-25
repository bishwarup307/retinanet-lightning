"""
__author__: bishwarup307
Created: 22/11/20
"""
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning import _logger as logger
from pytorch_lightning.utilities import rank_zero_only

from coco.cocoeval import COCOeval
from retinanet.anchors import MultiBoxPrior
from retinanet.losses import FocalLoss
from retinanet.utils import (
    ccwh_to_xyxy,
    ifnone,
    xyxy_to_ccwh,
    load_obj,
    get_total_steps,
    to_tensor,
    all_gather,
    coco_to_preds,
)


has_apex = False
try:
    import apex

    has_apex = True
except ModuleNotFoundError:
    pass


class RetinaNet(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        anchors: Optional[torch.Tensor] = None,
        dataset_val: Optional[Any] = None,
        dataset_size: Optional[int] = 1,
    ):
        super(RetinaNet, self).__init__()
        self.save_hyperparameters("cfg")
        self.backbone = _get_backbone(cfg.Model.backbone.name, pretrained=cfg.Model.backbone.pretrained)
        if not cfg.Model.backbone.pretrained:
            self.backbone.apply(init_weights)
        self.num_classes = cfg.Model.head.classification.num_classes
        fmap_sizes = _get_feature_depths(str(self.backbone))
        self.fpn = FPN(fmap_sizes, cfg.Model.FPN.channels, cfg.Model.FPN.upsample)
        num_priors = len(cfg.Model.anchors.scales) * len(cfg.Model.anchors.ratios)
        self.classification = RetineNetHead(
            self.num_classes * num_priors,
            cfg.Model.FPN.channels,
            use_bn=cfg.Model.head.classification.use_bn,
            num_repeats=cfg.Model.head.classification.n_repeat,
            activation=cfg.Model.head.classification.activation,
        )

        self.regression = RetineNetHead(
            num_priors * 4,
            cfg.Model.FPN.channels,
            use_bn=cfg.Model.head.regression.use_bn,
            num_repeats=cfg.Model.head.regression.n_repeat,
            activation=cfg.Model.head.regression.activation,
        )

        if cfg.Model.backbone.freeze_bn:
            self.backbone.freeze_bn()

        # self.classification_loss = nn.BCEWithLogitsLoss()
        self.classification_loss = FocalLoss(
            alpha=cfg.Model.head.classification.loss.params.alpha,
            gamma=cfg.Model.head.classification.loss.params.gamma,
            reduction="sum",
        )
        self.regression_loss = nn.SmoothL1Loss(reduction="mean", beta=cfg.Model.head.regression.loss.params.beta)
        self.calculate_bbox = OffsetsToBBox()
        self.nms = NMS()
        # self.anchors = nn.Parameter(anchors, requires_grad=False)

        # get anchors
        if anchors is None:
            logger.info("Model initialized without anchors.")
        # anchors = ifnone(anchors, torch.empty([]))

        if anchors is None:
            m = MultiBoxPrior()
            sample_image = torch.randn(
                size=(4, 3, cfg.Dataset.image_size[0], cfg.Dataset.image_size[1]),
                device=self.device,
            )
            anchors = m(sample_image)

        self.register_buffer("anchors", anchors)

        self.image_size = cfg.Dataset.image_size[:2]

        if dataset_val is not None:
            self.dataset_val = dataset_val
            self.coco_labels = dataset_val.coco_labels
            self.val_coco_gt = dataset_val.coco
        else:
            self.dataset_val = None
            self.coco_labels = None
            self.val_coco_gt = None

        self.coco_label_map = None
        self.test_predictions_name = None
        self.model_name = None

        self.val_preds = "val" if cfg.Trainer.save_val_predictions else None
        self.test_preds = "test" if cfg.Trainer.save_test_predictions else None
        self.optimizer = cfg.Trainer.optimizer
        self.scheduler = cfg.Trainer.scheduler
        self.total_steps = get_total_steps(dataset_size, cfg.Trainer.batch_size.train) * cfg.Trainer.num_epochs

        prior_mean = ifnone(cfg.Model.anchors.prior_mean, [0, 0, 0, 0])
        prior_std = ifnone(cfg.Model.anchors.prior_std, [0.1, 0.1, 0.2, 0.2])
        self.register_buffer("prior_mean", to_tensor(prior_mean))
        self.register_buffer("prior_std", to_tensor(prior_std))

        self.effective_batch_size = cfg.Trainer.batch_size.train * cfg.Trainer.gpus

        self.fpn.apply(init_weights)
        self.classification.apply(init_weights)
        self.regression.apply(init_weights)

        self.classification.block[-1].weight.data.fill_(0)
        self.classification.block[-1].bias.data.fill_(
            -math.log((1.0 - cfg.Model.head.classification.bias_prior) / cfg.Model.head.classification.bias_prior)
        )
        self.regression.block[-1].weight.data.fill_(0)
        self.regression.block[-1].bias.data.fill_(0)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, os.PathLike],
        config_path: Optional[Union[str, os.PathLike]] = None,
        name: Optional[str] = None,
    ):

        if config_path is None:
            logdir = Path(checkpoint_path).parent.parent
        else:
            logdir = Path(config_path)
        cfg_path = logdir.joinpath("hparams.yaml")

        try:
            cfg = OmegaConf.load(cfg_path)
        except FileNotFoundError:
            raise ValueError(f"could not find `hparams.yaml` in {str(cfg_path)}. ")

        try:
            ckpt = torch.load(checkpoint_path)
        except FileNotFoundError:
            raise ValueError(f"Could not load checkpoint {checkpoint_path}")

        label_map_file = logdir.joinpath("coco_label_map.json")
        if label_map_file.is_file():
            with open(label_map_file, "r") as f:
                coco_label_map = json.load(f)

        coco_labels_file = logdir.joinpath("coco_labels.json")
        if coco_labels_file.is_file():
            with open(coco_labels_file, "r") as f:
                coco_labels = json.load(f)

        model = cls(cfg.cfg, anchors=ckpt["state_dict"]["anchors"])
        model.coco_labels = model.cast_keys_ints(coco_labels)
        model.coco_label_map = model.cast_keys_ints(coco_label_map)
        model.load_state_dict(ckpt["state_dict"])
        model.test_predictions_name = name
        model.model_name = Path(checkpoint_path).stem
        return model

    def _forward_classification_head(self, logits: torch.Tensor, gt_cls: torch.Tensor) -> torch.Tensor:
        num_pos = (gt_cls > 0).sum(axis=1)
        num_assigned = (gt_cls > -1).sum(axis=1)

        mask = gt_cls > -1
        masked_gt = gt_cls[mask]
        masked_logits = logits[mask]
        masked_target_one_hot = F.one_hot(
            masked_gt.long(), num_classes=self.num_classes + 1  # +1 for background
        ).float()
        # background class is not used for loss calculation
        # https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/meta_arch/retinanet.py#L321
        cls_loss = self.classification_loss(masked_logits, masked_target_one_hot[:, 1:], num_pos, num_assigned)
        return cls_loss

    def _forward_regression_head(
        self,
        pred_box_offsets: torch.Tensor,
        gt_box_offsets: torch.Tensor,
        gt_cls: torch.Tensor,
    ) -> torch.Tensor:
        mask = gt_cls > 0
        masked_gt = gt_box_offsets[mask]
        masked_offsets = pred_box_offsets[mask]
        reg_loss = self.regression_loss(masked_offsets, masked_gt)
        return reg_loss

    def forward(self, t: torch.Tensor):
        features = self.backbone(t)
        pyramids = self.fpn(*features)

        logits, offsets = [], []
        for feature in pyramids:
            cls_pred = self.classification(feature)
            reg_pred = self.regression(feature)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(t.size(0), -1, self.num_classes)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(t.size(0), -1, 4)
            logits.append(cls_pred)
            offsets.append(reg_pred)
        return torch.cat(logits, dim=1), torch.cat(offsets, dim=1)

    def on_train_start(self) -> None:
        self._save_dataset_attributes()

    def on_train_epoch_start(self) -> None:
        self.backbone.freeze_bn()

    def on_validation_epoch_start(self) -> None:
        self._initialize_trackers()

    def on_test_epoch_start(self) -> None:
        self._initialize_trackers()

    def training_step(self, batch, batch_idx):
        img, gt_boxes, gt_cls = batch
        logits, offsets = self(img)

        # calculate classification loss
        cls_loss = self._forward_classification_head(logits, gt_cls)
        # calculate regression loss
        reg_loss = self._forward_regression_head(offsets, gt_boxes, gt_cls)

        loss = cls_loss + reg_loss

        self.log("train/cls_loss", cls_loss, prog_bar=True)
        self.log("train/reg_loss", reg_loss, prog_bar=True)
        self.log("loss", loss, logger=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        img, gt_boxes, gt_cls, scales, offset_x, offset_y, image_ids = batch
        (
            cls_loss,
            reg_loss,
            nms_image_idx,
            nms_bboxes,
            nms_classes,
            nms_scores,
        ) = self._nms_forward(img, gt_boxes, gt_cls)

        self._update_trackers(
            image_ids[nms_image_idx],
            nms_bboxes,
            nms_scores,
            nms_classes,
            scales[nms_image_idx],
            offset_x[nms_image_idx],
            offset_y[nms_image_idx],
        )

        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
        }

    def test_step(self, batch, batch_idx):
        try:
            img, gt_boxes, gt_cls, scales, offset_x, offset_y, image_ids = batch
        except ValueError:
            img, scales, offset_x, offset_y, image_ids = batch
            gt_boxes, gt_cls = None, None

        (
            cls_loss,
            reg_loss,
            nms_image_idx,
            nms_bboxes,
            nms_classes,
            nms_scores,
        ) = self._nms_forward(img, gt_boxes, gt_cls)

        self._update_trackers(
            image_ids[nms_image_idx],
            nms_bboxes,
            nms_scores,
            nms_classes,
            scales[nms_image_idx],
            offset_x[nms_image_idx],
            offset_y[nms_image_idx],
        )

        # metrics = self.validation_step(batch, batch_idx)
        metrics = {"cls_loss": cls_loss, "reg_loss": reg_loss}
        return metrics

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_cls_loss = torch.stack([x["cls_loss"] for x in outputs])
        avg_reg_loss = torch.stack([x["reg_loss"] for x in outputs])
        avg_cls_loss = avg_cls_loss[~torch.isnan(avg_cls_loss)].mean()
        avg_reg_loss = avg_cls_loss[
            ~torch.isnan(avg_reg_loss)
        ].mean()  # a batch with no annotation will likely result in nan reg_loss

        val_loss = avg_cls_loss + avg_reg_loss

        metrics = {
            "val/cls_loss": avg_cls_loss,
            "val/reg_loss": avg_reg_loss,
            "val_loss": val_loss,
        }
        self.log_dict(metrics, sync_dist=True)
        self._adjust_scales_offsets()
        self._sync_processes()
        stats = self.coco_eval(save_name=self.val_preds)
        self._log_coco_results(stats)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        try:
            avg_cls_loss = torch.stack([x["cls_loss"] for x in outputs])
            avg_reg_loss = torch.stack([x["reg_loss"] for x in outputs])
            avg_cls_loss = avg_cls_loss[~torch.isnan(avg_cls_loss)].mean()
            avg_reg_loss = avg_cls_loss[~torch.isnan(avg_reg_loss)].mean()
        except TypeError:
            avg_cls_loss = -1
            avg_reg_loss = -1

        metrics = {
            "test/cls_loss": avg_cls_loss,
            "test/reg_loss": avg_reg_loss,
        }
        self.log_dict(metrics, sync_dist=True)
        self._adjust_scales_offsets()
        self._sync_processes()

        do_coco_eval = self.val_coco_gt is not None and self.coco_labels is not None and self.dataset_val is not None
        if do_coco_eval:
            stats = self.coco_eval(save_name=self.test_preds)
            self._log_coco_results(stats, stage="test")
        else:
            self.save_test_predictions()

    #         logger.info(f"test loss(cls): {avg_cls_loss:.4f}")
    #         logger.info(f"test loss(reg): {avg_reg_loss:.4f}")

    def _sync_processes(self):
        boxes = self.boxes.detach().cpu().numpy()
        image_ids = self.image_ids.detach().cpu().numpy()
        scores = self.scores.detach().cpu().numpy()
        class_ids = self.class_idx.detach().cpu().numpy()

        bb_sync = all_gather(boxes)
        imageid_sync = all_gather(image_ids)
        scores_sync = all_gather(scores)
        classids_sync = all_gather(class_ids)

        k = zip(
            np.concatenate(bb_sync).tolist(),
            np.concatenate(imageid_sync).tolist(),
            np.concatenate(scores_sync).tolist(),
            list(map(self.coco_labels.get, np.concatenate(classids_sync).tolist())),
        )
        coco_res = [dict(zip(["bbox", "image_id", "score", "category_id"], p)) for p in k]
        self.coco_res = coco_res

    @rank_zero_only
    def coco_eval(self, save_name: Optional[str] = None):
        gt = copy.deepcopy(self.val_coco_gt)
        dt = gt.loadRes(self.coco_res)
        if save_name is not None:
            predictions = coco_to_preds(dt)
            with open(Path(self.logger.log_dir).joinpath(f"{save_name}_predictions.json"), "w") as f:
                json.dump(predictions, f, indent=2)
        coco_eval = COCOeval(gt, dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats

    @rank_zero_only
    def _save_dataset_attributes(self):
        with open(Path(self.logger.log_dir).joinpath("coco_labels.json"), "w") as f:
            json.dump(self.coco_labels, f, indent=2)
        with open(Path(self.logger.log_dir).joinpath("coco_label_map.json"), "w") as f:
            json.dump(self.dataset_val.coco_label_map, f, indent=2)

    @rank_zero_only
    def save_test_predictions(self):
        if self.coco_label_map is None:
            raise ValueError("model does not have a `coco_label_map`")
        preds = dict()
        for det in self.coco_res:
            key = Path(self.test_dataloader.dataloader.dataset.images[det["image_id"]]).name
            value = {
                "bbox": list(map(int, det["bbox"])),
                "confidence": det["score"],
                "class_index": self.coco_label_map[det["category_id"]],
            }
            if key in preds:
                preds[key].append(value)
            else:
                preds[key] = [value]

        prediction_name = ifnone(self.test_predictions_name, self.model_name + ".json")
        save_path = Path(self.logger.log_dir).joinpath(prediction_name)
        with open(save_path, "w") as f:
            json.dump(preds, f, indent=2)
        logger.info(f"predictions saved in {str(save_path)}")

    def _initialize_trackers(self) -> None:
        self.image_ids = torch.tensor([], device=self.device, dtype=torch.long)
        self.boxes = torch.tensor([], device=self.device)
        self.scores = torch.tensor([], device=self.device)
        self.class_idx = torch.tensor([], device=self.device, dtype=torch.long)
        self.scales = torch.tensor([], device=self.device)
        self.offset_x = torch.tensor([], device=self.device)
        self.offset_y = torch.tensor([], device=self.device)

    def _update_trackers(
        self,
        image_ids: torch.Tensor,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        scales: torch.Tensor,
        offset_x: torch.Tensor,
        offset_y: torch.Tensor,
    ) -> None:
        self.image_ids = torch.cat([self.image_ids, image_ids])
        self.boxes = torch.cat([self.boxes, boxes])
        self.scores = torch.cat([self.scores, scores])
        self.class_idx = torch.cat([self.class_idx, class_ids])
        self.scales = torch.cat([self.scales, scales])
        self.offset_x = torch.cat([self.offset_x, offset_x])
        self.offset_y = torch.cat([self.offset_y, offset_y])

    def _nms_forward(self, img: torch.Tensor, gt_boxes: torch.Tensor, gt_cls: torch.Tensor):
        # img, gt_boxes, gt_cls, scales, offset_x, offset_y, image_ids = batch
        logits, offsets = self(img)

        pred_boxes = self.calculate_bbox(self.anchors, offsets, self.image_size, self.prior_mean, self.prior_std)
        nms_image_idx, nms_bboxes, nms_classes, nms_scores = self.nms(torch.sigmoid(logits), pred_boxes)

        calc_loss = gt_cls is not None and gt_boxes is not None
        cls_loss, reg_loss = None, None
        if calc_loss:
            cls_loss = self._forward_classification_head(logits, gt_cls)
            reg_loss = self._forward_regression_head(offsets, gt_boxes, gt_cls)

        return cls_loss, reg_loss, nms_image_idx, nms_bboxes, nms_classes, nms_scores

    def _adjust_scales_offsets(self) -> None:
        self.boxes[:, 2] = self.boxes[:, 2] - self.boxes[:, 0]
        self.boxes[:, 3] = self.boxes[:, 3] - self.boxes[:, 1]
        self.boxes[:, 0] = self.boxes[:, 0] - self.offset_x
        self.boxes[:, 1] = self.boxes[:, 1] - self.offset_y
        self.boxes = self.boxes / self.scales[:, None]

    @rank_zero_only
    def _log_coco_results(self, stats, stage="eval"):
        map_avg, map_50, map_75, map_small, map_medium, map_large, *_ = stats
        self.log(f"COCO_{stage}/mAP@0.5:0.95:0.05", map_avg)
        self.log(f"COCO_{stage}/mAP@0.5", map_50)
        self.log(f"COCO_{stage}/mAP@0.75", map_75)
        self.log(f"COCO_{stage}/mAP_small", map_small)
        self.log(f"COCO_{stage}/mAP_medium", map_medium)
        self.log(f"COCO_{stage}/mAP_large", map_large)

    @property
    def parameter_count(self):
        count = sum([p.numel() for p in self.parameters()])
        return f"{count:,}"

    @staticmethod
    def cast_keys_ints(d: Dict):
        new_dict = dict()
        for k, v in d.items():
            new_dict[int(k)] = v
        return new_dict

    def configure_optimizers(self):
        if self.effective_batch_size > 64 and has_apex:
            logger.info("Training with LAMB optimizer")
            optimizer = apex.optimizers.FusedLAMB(self.parameters(), lr=1e-3, weight_decay=0.01)
        else:
            optimizer = ifnone(
                load_obj(self.optimizer.name)(self.parameters(), **self.optimizer.params),
                torch.optim.Adam(self.parameters(), lr=1e-5),
            )
            logger.info("Optimizer: adam")

        #         optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        scheduler = load_obj(self.scheduler.name)(optimizer, total_steps=self.total_steps, **self.scheduler.params)
        schedulers = {
            "scheduler": scheduler,
            "interval": "step",
        }
        return [optimizer], [schedulers]


class RetineNetHead(nn.Module):
    def __init__(
        self,
        out_channels: int,
        channels: Optional[int] = 256,
        use_bn: Optional[bool] = False,
        num_repeats: Optional[int] = 4,
        activation: Optional[str] = "relu",
    ):
        super(RetineNetHead, self).__init__()
        blocks = []
        for _ in range(num_repeats):
            blocks.append(ConvBNRelu(channels, channels, use_bn=use_bn, act=activation))
        blocks.append(nn.Conv2d(channels, out_channels, 3, 1, 1))
        self.block = nn.Sequential(*blocks)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.block(t)


class ResNet(nn.Module):
    """
    Implements ResNet backbone for retinanet.
    Args:
        depth (int): depth for resnet, one of 18, 34, 50, 101, 152
        pretrained (bool): whether to load pretrained ImageNet weights
    """

    def __init__(self, depth: int, pretrained: bool = False):

        if depth not in (18, 34, 50, 101, 152):
            raise ValueError(f"invalid depth specified. depth must be one of 18, 34, 50, 101, 152, got {depth}")

        super(ResNet, self).__init__()
        self.model_repr = f"resnet{depth}"
        self.pretrained = pretrained
        self._register_layers()

    def __repr__(self):
        return self.model_repr

    def _register_layers(self):
        base = torchvision.models.__dict__[self.model_repr](pretrained=self.pretrained)
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.convs = nn.ModuleList([base.layer1, base.layer2, base.layer3, base.layer4])

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x: torch.Tensor):
        features = []
        feat = self.layer0(x)

        for i, conv in enumerate(self.convs):
            feat = conv(feat)
            if i > 0:
                features.append(feat)
        return features


class ResNeXt(ResNet, nn.Module):
    """
    Implements ResNeXt backbone for retinanet.
    Args:
        depth (int): depth for resnet, either 50 or 101
        pretrained (bool): whether to load pretrained ImageNet weights
    """

    def __init__(self, depth: int, pretrained: bool = False):
        if depth not in (50, 101):
            raise ValueError(f"invalid depth specified. depth must be one of 50, 101 got {depth}")

        super(ResNet, self).__init__()
        self.model_repr = f"resnext{depth}_32x{4 * (depth // 50)}d"
        self.pretrained = pretrained
        self._register_layers()


class WideResNet(ResNet, nn.Module):
    """
    Implements Wide ResNet backbone for retinanet.
    Args:
        depth (int): depth for resnet, either 50 or 101
        pretrained (bool): whether to load pretrained ImageNet weights
    """

    def __init__(self, depth: int, pretrained: bool = False):
        if depth not in (50, 101):
            raise ValueError(f"invalid depth specified. depth must be one of 50, 101 got {depth}")

        super(ResNet, self).__init__()
        self.model_repr = f"wide_resnet{depth}_2"
        self.pretrained = pretrained
        self._register_layers()


class Conv_1x1(nn.Module):
    """
    1x1 conv block
    """

    def __init__(self, in_ch: int, out_ch: int):
        super(Conv_1x1, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        return self.conv(x)


class ConvBNRelu(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Optional[int] = 3,
        stride: Optional[int] = 1,
        padding: Optional[int] = 1,
        act: Optional[str] = None,
        use_bn: Optional[bool] = False,
        **kwargs,
    ):
        super(ConvBNRelu, self).__init__()
        if act is None or act == "relu":
            act = nn.ReLU()
        elif act == "leakyrelu":
            act = nn.LeakyReLU(0.2)
        else:
            raise NotImplementedError
        ops = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, **kwargs)]
        if use_bn:
            ops.append(nn.BatchNorm2d(out_ch))
        ops.append(act)

        self.block = nn.Sequential(*ops)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.block(t)


class UpsampleSkipBlock(nn.Module):
    """
    Upsample and merge with skip connection block.
                upsample 2x
                -----------
                    ↓
        --> 1x1 ->  ▣
                    ↓
                -----------
                   3x3
                -----------
                    ↓

    Upsample the top level feature with stride 2 to merge with the corresponding lateral (skip)
    features from the bottom-up pathway. finally applies a 3x3 conv mainly for smoothing the
    aliasing (or checkerboard if ConvTranspose2d is used https://distill.pub/2016/deconv-checkerboard/)

    Args:
        in_channels (int): incoming channels from top
        lateral_channels (int): incoming channels from skip connection
        upsample (Optional[str]): whether to use upsampling. choices are `nearest` or `bilinear`. ConvTranspose2d is
        used instead if `None`. Default is `None`.
    Returns:
        (torch.Tensor): merged feature map from top and lateral branches.
    """

    def __init__(self, in_channels: int, lateral_channels: int, upsample: Optional[str] = None):
        super(UpsampleSkipBlock, self).__init__()
        if upsample is not None:
            assert upsample in (
                "nearest",
                "bilinear",
            ), f"Upsample mode must be either `nearest` or `bilinear`, got {upsample}"
            align_corners = False if upsample == "bilinear" else None
            self.up = nn.Upsample(scale_factor=2, mode=upsample, align_corners=align_corners)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=2,
                padding=0,
                stride=2,
            )
        self.skip = Conv_1x1(lateral_channels, in_channels)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, top: torch.Tensor, skip: torch.Tensor):
        merged = self.up(top)
        merged = merged + self.skip(skip)
        merged_smoothed = self.conv(merged)
        return merged_smoothed, merged


class FPN(nn.Module):
    """
    Implements Feature Pyramid Networks for Object Detection (https://arxiv.org/abs/1612.03144).
    This implementation is as provided in the retinanet paper (https://arxiv.org/abs/1708.02002) and hence
    differ from the original FPN paper. In particular, quoting the paper:

    ##############################################################################################
    RetinaNet uses feature pyramid levels P3 to P7, where P3 to P5 are computed from the output of
    the corresponding ResNet residual stage (C3 through C5) using top-down and lateral connections
    just as in [20], P6 is obtained via a 3×3 stride-2 conv on C5, and P7 is computed by apply- ing ReLU
    followed by a 3×3 stride-2 conv on P6. This differs slightly from [20]:
    (1) we don’t use the high-resolution pyramid level P2 for com- putational reasons,
    (2) P6 is computed by strided convolution instead of downsampling,
    and (3) we include P7 to improve large object detection. These minor modifications improve
    speed while maintaining accuracy.
    ##############################################################################################

    Args:
        in_channels (Sequence[int]): Channels from the bottom up backbone. e.g. for ResNet based backbones
        depth of features from layer2, layer3 and layer4.
        channels (int): dimension of the top down channels. Default is 256 as in the retinanet paper.
        upsample (Optional[str]): whether to use upsampling. choices are `nearest` or `bilinear`. ConvTranspose2d is
        used instead if `None`. Default is `None`.
    Returns:
        List[Torch.Tensor]: [P3, P4, P5, P6, P7] where P represents pyramid features at various levels.
    """

    def __init__(
        self,
        in_channels: Tuple[int, int, int],
        channels: int = 256,
        upsample: Optional[str] = None,
    ):
        super(FPN, self).__init__()
        ch_c3, ch_c4, ch_c5 = in_channels

        self.p5_1 = Conv_1x1(ch_c5, channels)
        self.p5_2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.p4 = UpsampleSkipBlock(channels, ch_c4, upsample)
        self.p3 = UpsampleSkipBlock(channels, ch_c3, upsample)
        self.p6 = nn.Conv2d(ch_c5, channels, kernel_size=3, stride=2, padding=1)
        self.p7 = nn.Sequential(nn.ReLU(), nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1))

    def forward(self, C3: torch.Tensor, C4: torch.Tensor, C5: torch.Tensor):
        P5_1 = self.p5_1(C5)
        P5 = self.p5_2(P5_1)
        P4, P4_1 = self.p4(P5_1, C4)
        P3, _ = self.p3(P4_1, C3)
        P6 = self.p6(C5)
        P7 = self.p7(P6)
        return P3, P4, P5, P6, P7


def _get_feature_depths(model_name: str) -> Tuple[int, int, int]:
    """
    Get depth of feature maps from various backbones.
    Args:
        model_name (str): model str representing the pretrained models.

    Returns:
        Tuple[int, int, int]: dimension of [C3, C4, C5].

    """
    feature_maps = {
        "resnet18": (128, 256, 512),
        "resnet34": (128, 256, 512),
        "resnet50": (512, 1024, 2048),
        "resnet101": (512, 1024, 2048),
        "resnet152": (512, 1024, 2048),
        "resnext50_32x4d": (512, 1024, 2048),
        "resnext101_32x8d": (512, 1024, 2048),
        "wide_resnet50_2": (512, 1024, 2048),
        "wide_resnet101_2": (512, 1024, 2048),
    }
    return feature_maps[model_name]


def _get_backbone(name: str, depth: Optional[int] = None, pretrained: Optional[bool] = False):
    backbones = {"resnet": ResNet, "resnext": ResNeXt, "wide_resnet": WideResNet}

    if depth is None:
        try:
            depth = int(name.split("_")[-1])
            name = "_".join([x.lower() for x in name.split("_")[:-1]])
        except Exception as e:
            print("Couldn't understand the specified backbone")
            raise e

    if name not in backbones:
        raise ValueError(f"Invalid backbone specified. Currently supported {','.join(list(backbones.keys()))}")

    return backbones[name](depth=depth, pretrained=pretrained)


class OffsetsToBBox(nn.Module):
    def __init__(self):
        """
        Transforms the predicted offsets (deltas) from the model to bounding box coordinates.
        Args:
            mean: prior mean. Defaults to [0, 0, 0, 0] for
            :math:`(offset_{x_{center}}, offset_{y_{center}}, offset_{width}, offset_{height})`
            std: prior variance. Defaults to [0.1, 0.1, 0.2, 0.2] for
            :math:`(offset_{x_{center}}, offset_{y_{center}}, offset_{width}, offset_{height})`
        """
        super(OffsetsToBBox, self).__init__()
        # self.mean = ifnone(
        #     mean,
        #     torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
        #     # nn.Parameter(
        #     #     torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)),
        #     #     requires_grad=False,
        #     # ),
        # )
        # self.std = ifnone(
        #     std,
        #     torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        #     # nn.Parameter(
        #     #     torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)),
        #     #     requires_grad=False,
        #     # ),
        # )

    def _clip_boxes(self, boxes, image_size: Sequence[int]):
        """
        clip bounding box coodinates between 0 and image width and height.
        """
        boxes[..., 0::2] = torch.clamp(boxes[..., 0::2], min=0, max=image_size[0])
        boxes[..., 1::2] = torch.clamp(boxes[..., 1::2], min=0, max=image_size[1])
        return boxes

    @torch.no_grad()
    def forward(
        self,
        anchors: torch.Tensor,
        offsets: torch.Tensor,
        image_size: Sequence[int],
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        anchors = xyxy_to_ccwh(anchors)
        offsets = offsets * std + mean

        cc = offsets[..., :2]
        wh = offsets[..., 2:]

        cc = cc * anchors[..., 2:] + anchors[..., :2]
        wh = wh.exp() * anchors[..., 2:]
        boxes_ccwh = torch.cat([cc, wh], dim=-1)
        boxes_xyxy = ccwh_to_xyxy(boxes_ccwh)
        return self._clip_boxes(boxes_xyxy, image_size)


class NMS(nn.Module):
    def __init__(self, conf_threshold: float = 0.1, nms_threshold: float = 0.5):
        super(NMS, self).__init__()
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def forward(self, logits: torch.Tensor, boxes: torch.Tensor):
        num_classes = logits.size(-1)
        batch_size = logits.size(0)
        # print(f"batched_nms >> num_classes: {num_classes}")
        # print(logits.size())
        scores, class_indices = logits.max(dim=2)
        rows = (torch.arange(batch_size, dtype=torch.long)[:, None] * num_classes).type_as(class_indices)
        cat_idx = class_indices + rows
        #         print(f"batched_nms >> max_score: {scores.max()}")
        #         print(f"batched_nms >> min_score: {scores.min()}")
        mask = scores > self.conf_threshold
        instances_per_image = mask.sum(dim=1)
        #         print(f"batched_nms >> instance per image: {instances_per_image}")
        #         selected_class_indices = class_indices[mask]
        image_ids = torch.arange(batch_size).type_as(class_indices)
        image_ids = torch.repeat_interleave(image_ids, instances_per_image)
        #         category_idx = image_ids * (selected_class_indices + 1)
        #         selected_bboxes = boxes[mask]

        # print(selected_bboxes.size())
        # print(category_idx.size())
        # print(scores[mask].size())

        keep_indices = torchvision.ops.boxes.batched_nms(boxes[mask], scores[mask], cat_idx[mask], self.nms_threshold)
        nms_image_idx = image_ids[keep_indices]
        nms_bboxes = boxes[mask][keep_indices]
        nms_scores = scores[mask][keep_indices]
        nms_classes = class_indices[mask][keep_indices]
        return nms_image_idx, nms_bboxes, nms_classes, nms_scores


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.zero_()


if __name__ == "__main__":
    input_t = torch.randn(4, 3, 256, 256)
    # backbone = _get_backbone("resnext_50")
    # backbone = ResNet(depth=50, pretrained=False)
    # fmap_channels = _get_feature_depths(str(backbone))
    # fpn = FPN(fmap_channels)
    # output_features = backbone(input_t)
    # pyramid_features = fpn(*output_features)
    retinanet = RetinaNet("resnet_34", 2)
    # for feat in output_features:
    #     print(feat.size())
    cls, loc = retinanet(input_t)
    print(cls.size())
    print(loc.size())
