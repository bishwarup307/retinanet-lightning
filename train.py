"""
__author__: bishwarup307
Created: 03/12/20
"""

import argparse

import pytorch_lightning as pl
from omegaconf import OmegaConf

from retinanet.datasets import DataModule
from retinanet.models import RetinaNet
from retinanet.utils import get_device_config, get_callbacks, ifnone, get_logger
from pytorch_lightning import _logger as logger
import torch.distributed as dist

# logger = get_logger(__name__)


def parse_cli_args():
    parser = argparse.ArgumentParser("Retinanet config")
    parser.add_argument("-cfg", "--config-path", type=str, default="./config.yaml")
    parser.add_argument("--root", type=str, required=False, help="dataset root directory")
    parser.add_argument("--logdir", type=str, required=False, help="root log directory")
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
        cfg=cfg,
        anchors=dm.anchors,
        dataset_val=dm.val_dataset,
        dataset_size=dm.train_samples,
    )

    logger.info("successfully initialized the model")

    gpus, tpus = get_device_config(cfg.Trainer.gpus, cfg.Trainer.tpus)
    callbacks = get_callbacks(cfg.Trainer.callbacks)

    accelerator = cfg.Trainer.dist_backend if gpus > 1 else None

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
        accelerator=accelerator,
    )

    trainer.fit(model, dm)

    if dm.test_dataset is not None:
        logger.info("starting test...")
        model.val_coco_gt = dm.test_dataset.coco
        trainer.test(datamodule=dm)


if __name__ == "__main__":
    main()
