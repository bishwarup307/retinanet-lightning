"""
__author__: bishwarup307
Created: 22/11/20
"""
import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import copy

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from retinanet.anchors import MultiBoxPrior
from retinanet.datasets import CocoDataset
from retinanet.losses import FocalLoss
from retinanet.utils import batched_nms, ccwh_to_xyxy, ifnone, xyxy_to_ccwh


class RetinaNet(pl.LightningModule):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained_backbone: bool = True,
        channels: int = 256,
        num_priors: int = 9,
        fpn_upsample: str = "bilinear",
        head_num_repeats: int = 4,
        head_activation: str = "relu",
        head_use_bn: bool = False,
        backbone_freeze_bn: bool = True,
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2.0,
        l1_loss_beta: float = 0.1,
        image_size: Optional[Tuple[int, int]] = None,
        anchors: Optional[torch.Tensor] = None,
        prior_mean: Optional[List[float]] = None,
        prior_std: Optional[List[float]] = None,
        coco_labels: Optional[Dict] = None,
        val_coco_gt: Optional[Any] = None,
        classification_bias_prior: float = 0.01,
    ):
        super(RetinaNet, self).__init__()
        self.backbone = _get_backbone(backbone, pretrained=pretrained_backbone)
        if not pretrained_backbone:
            self.backbone.apply(init_weights)
        self.num_classes = num_classes
        fmap_sizes = _get_feature_depths(str(self.backbone))
        self.fpn = FPN(fmap_sizes, channels, fpn_upsample)
        self.classification = RetineNetHead(
            num_classes * num_priors,
            channels,
            use_bn=head_use_bn,
            num_repeats=head_num_repeats,
            activation=head_activation,
        )

        self.regression = RetineNetHead(
            num_priors * 4,
            channels,
            use_bn=head_use_bn,
            num_repeats=head_num_repeats,
            activation=head_activation,
        )

        if backbone_freeze_bn:
            self.backbone.freeze_bn()

        # self.classification_loss = nn.BCEWithLogitsLoss()
        self.classification_loss = FocalLoss(
            alpha=focal_loss_alpha, gamma=focal_loss_gamma, reduction="sum"
        )
        self.regression_loss = nn.SmoothL1Loss(reduction="mean", beta=l1_loss_beta)
        self.calculate_bbox = OffsetsToBBox()
        self.nms = NMS()
        self.anchors = nn.Parameter(anchors, requires_grad=False)
        self.image_size = image_size
        self.coco_labels = coco_labels
        self.val_coco_gt = val_coco_gt
        self.prior_mean = ifnone(
            prior_mean,
            nn.Parameter(
                torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)),
                requires_grad=False,
            ),
        )
        self.prior_std = ifnone(
            prior_std,
            nn.Parameter(
                torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)),
                requires_grad=False,
            ),
        )

        self.fpn.apply(init_weights)
        self.classification.apply(init_weights)
        self.regression.apply(init_weights)

        self.classification.block[-1].weight.data.fill_(0)
        self.classification.block[-1].bias.data.fill_(
            -math.log((1.0 - classification_bias_prior) / classification_bias_prior)
        )
        self.regression.block[-1].weight.data.fill_(0)
        self.regression.block[-1].bias.data.fill_(0)

        # TODO: save all hparams except anchors
        # self.save_hyperparameters(self._hyperparams())

    def _forward_classification_head(
        self, logits: torch.Tensor, gt_cls: torch.Tensor
    ) -> torch.Tensor:
        num_pos = (gt_cls > 0).sum()
        mask = gt_cls > -1
        masked_gt = gt_cls[mask]
        masked_logits = logits[mask]
        masked_target_one_hot = F.one_hot(
            masked_gt.long(), num_classes=self.num_classes + 1  # +1 for background
        ).float()
        # background class is not used for loss calculation
        # https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/meta_arch/retinanet.py#L321
        cls_loss = (
            self.classification_loss(masked_logits, masked_target_one_hot[:, 1:])
            / num_pos
        )
        return cls_loss

    # def on_train_start(self):
    #     loader = self.train_dataloader()
    #     self.anchors = loader.dataset.anchors
    #     self.image_size = loader.dataset.image_size

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
            cls_pred = (
                cls_pred.permute(0, 2, 3, 1)
                .contiguous()
                .view(t.size(0), -1, self.num_classes)
            )
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(t.size(0), -1, 4)
            logits.append(cls_pred)
            offsets.append(reg_pred)
        return torch.cat(logits, dim=1), torch.cat(offsets, dim=1)
    
    def on_train_epoch_start(self) -> None:
        self.backbone.freeze_bn()

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
        self.log("loss", loss)
        return {"loss": loss}

    def _format_coco_results(
        self,
        image_ids,
        scales,
        offset_x,
        offset_y,
        nms_image_idx,
        nms_boxes,
        nms_classes,
        nms_scores,
    ):
        image_ids = image_ids.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        offset_x = offset_x.detach().cpu().numpy()
        offset_y = offset_y.detach().cpu().numpy()
        nms_image_idx = nms_image_idx.detach().cpu().numpy()
        nms_boxes = nms_boxes.detach().cpu().numpy()
        nms_classes = nms_classes.detach().cpu().numpy()
        nms_scores = nms_scores.detach().cpu().numpy()

        coco_results = []
        # transform to COCO coords
        nms_boxes[:, 2] -= nms_boxes[:, 0]
        nms_boxes[:, 3] -= nms_boxes[:, 1]

        for j, idx in enumerate(nms_image_idx):
            # j ->  across predicted results
            # idx -> across supplied ground truths
            imid = image_ids[idx]
            scale = scales[idx]
            ox, oy = offset_x[idx], offset_y[idx]
            score = nms_scores[j]
            class_index = nms_classes[j]
            bbox = nms_boxes[j]
            bbox[0] -= ox
            bbox[1] -= oy
            bbox = bbox / scale

            coco_res = {
                "image_id": imid.item(),
                "category_id": self.coco_labels[class_index],
                "score": float(score),
                "bbox": bbox.tolist(),
            }
            coco_results.append(coco_res)
        return coco_results

    def on_validation_epoch_start(self) -> None:
        self.image_ids = torch.tensor([], device=self.device, dtype=torch.long)
        self.boxes = torch.tensor([], device=self.device)
        self.scores = torch.tensor([], device=self.device)
        self.class_idx = torch.tensor([], device=self.device, dtype=torch.long)
        self.scales = torch.tensor([], device=self.device)
        self.offset_x = torch.tensor([], device=self.device)
        self.offset_y = torch.tensor([], device=self.device)

    def validation_step(self, batch, batch_idx):
        img, gt_boxes, gt_cls, scales, offset_x, offset_y, image_ids = batch

        logits, offsets = self(img)

        cls_loss = self._forward_classification_head(logits, gt_cls)
        reg_loss = self._forward_regression_head(offsets, gt_boxes, gt_cls)

        pred_boxes = self.calculate_bbox(
            self.anchors, offsets, self.image_size, self.prior_mean, self.prior_std
        )
        nms_image_idx, nms_bboxes, nms_classes, nms_scores = self.nms(
            torch.sigmoid(logits), pred_boxes
        )

        self.image_ids = torch.cat([self.image_ids, image_ids[nms_image_idx]])
        self.boxes = torch.cat([self.boxes, nms_bboxes])
        self.scores = torch.cat([self.scores, nms_scores])
        self.class_idx = torch.cat([self.class_idx, nms_classes])
        self.scales = torch.cat([self.scales, scales[nms_image_idx]])
        self.offset_x = torch.cat([self.offset_x, offset_x[nms_image_idx]])
        self.offset_y = torch.cat([self.offset_y, offset_y[nms_image_idx]])

        #         # print(f"nms_out: {nms_image_idx}")

        #         coco_res = self._format_coco_results(
        #             image_ids,
        #             scales,
        #             offset_x,
        #             offset_y,
        #             nms_image_idx,
        #             nms_bboxes,
        #             nms_classes,
        #             nms_scores,
        #         )
        #         # print(coco_res)
        #         coco_results.extend(coco_res)
        print(f"val_cls_loss: {cls_loss:.4f}, val_reg_loss: {reg_loss:.4f}")
        
        return {
            #             "val_loss": val_loss,
            "val_cls_loss": cls_loss,
            "val_reg_loss": reg_loss,
            #             "coco_results": coco_results,
        }
        

    def validation_epoch_end(self, outputs: List[Any]) -> None:
#         print(outputs)
        avg_cls_loss = torch.stack([x["val_cls_loss"] for x in outputs]).mean()
        avg_reg_loss = torch.stack([x["val_reg_loss"] for x in outputs]).mean()

        val_loss = avg_cls_loss + avg_reg_loss
        self.log("val/cls_loss", avg_cls_loss)
        self.log("val/reg_loss", avg_reg_loss)
        self.log("val_loss", val_loss)

        self.boxes[:, 2] = self.boxes[:, 2] - self.boxes[:, 0]
        self.boxes[:, 3] = self.boxes[:, 3] - self.boxes[:, 1]
        self.boxes[:, 0] = self.boxes[:, 0] - self.offset_x
        self.boxes[:, 1] = self.boxes[:, 1] - self.offset_y
        self.boxes = self.boxes / self.scales[:, None]
        self.coco_eval()
        
    def coco_eval(self):
        boxes = self.boxes.detach().cpu().numpy()
        image_ids = self.image_ids.detach().cpu().numpy()
        scores = self.scores.detach().cpu().numpy()
        class_ids = self.class_idx.detach().cpu().numpy()
        
        k = zip(boxes.tolist(), image_ids.tolist(), scores.tolist(), list(map(self.coco_labels.get, class_ids.tolist())))
        coco_res = [dict(zip(["bbox", "image_id", "score", "category_id"], p)) for p in k]

        gt = copy.deepcopy(self.val_coco_gt)
        dt = gt.loadRes(coco_res)
        cocoEval = COCOeval(gt,dt,"bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        
    #         print(f"validation-loss (classif): {avg_cls_loss.item(): .4f}")
    #         print(f"validation-loss (reg): {avg_reg_loss.item(): .4f}")

    #         with open("/home/ubuntu/val_bbox.json", "w") as f:
    #             json.dump(coco_results, f, indent=2)

    #         # print(coco_results)
    #         coco_dt = self.val_coco_gt.loadRes(coco_results)
    #         cocoEval = COCOeval(self.val_coco_gt, coco_dt, "bbox")
    #         cocoEval.evaluate()
    #         cocoEval.accumulate()
    #         cocoEval.summarize()

    def configure_optimizers(
        self,
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


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
            raise ValueError(
                f"invalid depth specified. depth must be one of 18, 34, 50, 101, 152, got {depth}"
            )

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
        # self._init_weights()

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
            raise ValueError(
                f"invalid depth specified. depth must be one of 50, 101 got {depth}"
            )

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
            raise ValueError(
                f"invalid depth specified. depth must be one of 50, 101 got {depth}"
            )

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

    def __init__(
        self, in_channels: int, lateral_channels: int, upsample: Optional[str] = None
    ):
        super(UpsampleSkipBlock, self).__init__()
        if upsample is not None:
            assert upsample in (
                "nearest",
                "bilinear",
            ), f"Upsample mode must be either `nearest` or `bilinear`, got {upsample}"
            align_corners = False if upsample == "bilinear" else None
            self.up = nn.Upsample(
                scale_factor=2, mode=upsample, align_corners=align_corners
            )
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
        merged = self.conv(merged)
        return merged


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
        self.p7 = nn.Sequential(
            nn.ReLU(), nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, C3: torch.Tensor, C4: torch.Tensor, C5: torch.Tensor):
        P5_1 = self.p5_1(C5)
        P5 = self.p5_2(P5_1)
        P4 = self.p4(P5_1, C4)
        P3 = self.p3(P4, C3)
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


def _get_backbone(
    name: str, depth: Optional[int] = None, pretrained: Optional[bool] = False
):
    backbones = {"resnet": ResNet, "resnext": ResNeXt, "wide_resnet": WideResNet}

    if depth is None:
        try:
            depth = int(name.split("_")[-1])
            name = "_".join([x.lower() for x in name.split("_")[:-1]])
        except Exception as e:
            print("Couldn't understand the specified backbone")
            raise e

    if name not in backbones:
        raise ValueError(
            f"Invalid backbone specified. Currently supported {','.join(list(backbones.keys()))}"
        )

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
#         offsets = offsets * std + mean

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
        rows = (
            torch.arange(batch_size, dtype=torch.long)[:, None] * num_classes
        ).type_as(class_indices)
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

        keep_indices = torchvision.ops.boxes.batched_nms(
            boxes[mask], scores[mask], cat_idx[mask], self.nms_threshold
        )
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
