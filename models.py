"""
__author__: bishwarup307
Created: 22/11/20
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl


class RetinaNet(pl.LightningModule):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        channels: Optional[int] = 256,
        num_priors: Optional[int] = 9,
        fpn_upsample: Optional[str] = "bilinear",
        head_num_repeats: Optional[int] = 4,
        head_activation: Optional[str] = "relu",
        head_use_bn: Optional[bool] = False,
        backbone_freeze_bn: Optional[bool] = True,
        focal_loss_alpha: Optional[float] = 0.25,
        focal_loss_gamma: Optional[float] = 2.0,
    ):
        super(RetinaNet, self).__init__()
        self.backbone = _get_backbone(backbone)
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
        depth [int]: depth for resnet, one of 18, 34, 50, 101, 152
        pretrained [bool]: whether to load pretrained ImageNet weights
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
        self._init_weights()

    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    torch.nn.init.kaiming_normal_(
                        m.weight, mode="fan_in", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.0)
                    if m.bias is not None:
                        m.bias.data.zero_()

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
        depth [int]: depth for resnet, either 50 or 101
        pretrained [bool]: whether to load pretrained ImageNet weights
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
        depth [int]: depth for resnet, either 50 or 101
        pretrained [bool]: whether to load pretrained ImageNet weights
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


class TopDownBlock(nn.Module):
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
        in_channels [int]: incoming channels from top
        lateral_channels [int]: incoming channels from skip connection
        upsample [Optional[str]]: whether to use upsampling. choices are `nearest` or `bilinear`. ConvTranspose2d is
        used instead if `None`. Default is `None`.
    Returns:
        [torch.Tensor]: merged feature map from top and lateral branches.
    """

    def __init__(
        self, in_channels: int, lateral_channels: int, upsample: Optional[str] = None
    ):
        super(TopDownBlock, self).__init__()
        if upsample is not None:
            assert upsample in (
                "nearest",
                "bilinear",
            ), f"Upsample mode must be either `nearest` or `bilinear`, got {upsample}"
            self.up = nn.Upsample(scale_factor=2, mode=upsample, align_corners=False)
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
        in_channels [Sequence[int]]: Channels from the bottom up backbone. e.g. for ResNet based backbones
        depth of features from layer2, layer3 and layer4.
        channels [int]: dimension of the top down channels. Default is 256 as in the retinanet paper.
        upsample [Optional[str]]: whether to use upsampling. choices are `nearest` or `bilinear`. ConvTranspose2d is
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

        self.p5 = Conv_1x1(ch_c5, channels)
        self.p4 = TopDownBlock(channels, ch_c4, upsample)
        self.p3 = TopDownBlock(channels, ch_c3, upsample)
        self.p6 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.p7 = nn.Sequential(
            nn.ReLU(), nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, C3: torch.Tensor, C4: torch.Tensor, C5: torch.Tensor):
        P5 = self.p5(C5)
        P4 = self.p4(P5, C4)
        P3 = self.p3(P4, C3)
        P6 = self.p6(P5)
        P7 = self.p7(P6)
        return P3, P4, P5, P6, P7


def _get_feature_depths(model_name: str) -> Tuple[int, int, int]:
    """
    Get depth of feature maps from various backbones.
    Args:
        model_name [str]: model str representing the pretrained models.

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
