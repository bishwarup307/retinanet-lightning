"""
__author__: bishwarup307
Created: 03/12/20
"""

from omegaconf import OmegaConf, DictConfig
import argparse
import pytorch_lightning as pl

from retinanet.datasets import DataModule
from retinanet.models import RetinaNet
from retinanet.utils import get_device_config, get_callbacks, ifnone, get_logger

logger = get_logger(__name__)


def parse_cli_args():
    parser = argparse.ArgumentParser("Retinanet config")
    parser.add_argument("-cfg", "--config-path", type=str, default="./config.yaml")
    parser.add_argument("--root", type=str, required=False)
    parser.add_argument("--logdir", type=str, required=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_cli_args()
    cfg = OmegaConf.load(args.config_path)

    cfg.Dataset.root = ifnone(args.root, cfg.Dataset.root)
    cfg.Trainer.logdir = ifnone(args.logdir, cfg.Trainer.logdir)

    dm = DataModule(cfg)
    dm.setup()
    logger.info("successfully loaded data module")

    model = RetinaNet(
        "resnet_34",
        num_classes=dm.num_classes,
        image_size=dm.image_size,
        anchors=dm.anchors,
        coco_labels=dm.val_dataset.coco_labels,
        val_coco_gt=dm.val_dataset.coco,
        pretrained_backbone=cfg.Model.backbone.pretrained,
        backbone_freeze_bn=cfg.Model.backbone.freeze_bn,
        channels=cfg.Model.FPN.channels,
        fpn_upsample=cfg.Model.FPN.upsample,
        focal_loss_alpha=cfg.Model.head.classification.loss.params.alpha,
        focal_loss_gamma=cfg.Model.head.classification.loss.params.gamma,
        l1_loss_beta=cfg.Model.head.regression.loss.params.beta,
        classification_head_num_repeats=cfg.Model.head.classification.n_repeat,
        classification_head_use_bn=cfg.Model.head.classification.use_bn,
        regression_head_num_repeats=cfg.Model.head.regression.n_repeat,
        regression_head_use_bn=cfg.Model.head.regression.use_bn,
        classification_bias_prior=cfg.Model.head.classification.classification_bias_prior,
        optimizer=cfg.Trainer.optimizer,
        scheduler=cfg.Trainer.scheduler,
        dataset_size=dm.train_samples,
        batch_size=cfg.Trainer.batch_size.train,
        epochs=cfg.Trainer.num_epochs,
    )
    logger.info("successfully initialized the model")

    gpus, tpus = get_device_config(cfg.Trainer.gpus, cfg.Trainer.tpus)
    callbacks = get_callbacks(cfg.Trainer.callbacks)

    logger.info("starting train...")
    trainer = pl.Trainer(
        default_root_dir=cfg.Trainer.logdir,
        max_epochs=cfg.Trainer.num_epochs,
        gpus=gpus,
        tpu_cores=tpus,
        precision=16 if cfg.Trainer.amp else 32,
        num_sanity_val_steps=cfg.Trainer.num_sanity_val_steps,
        gradient_clip_val=cfg.Trainer.clip_grad_norm,
        callbacks=callbacks,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
