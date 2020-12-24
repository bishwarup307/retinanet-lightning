"""
__author__: bishwarup307
Created: 09/12/20
"""

import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from retinanet.datasets import TestDataset
from retinanet.models import RetinaNet


def parse_args():
    parser = argparse.ArgumentParser(description="retinanet inference")
    parser.add_argument("-i", "--root", type=str, help="directory with test images")
    parser.add_argument("-s", "--image-size", type=str, default="512", help="image size for inference h,w")
    parser.add_argument("-w", "--weights", type=str, help="path to trained checkpoint")
    parser.add_argument("-bs", "--batch-size", type=int, default=8, help="batch size for inference")
    parser.add_argument("-o", "--output-dir", type=str, help="output directory to save the predictions and logs")
    parser.add_argument("--name", type=str, default=None, help="name of the output file")
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--workers", type=int, default=4, help="number of workers to use for dataloaders")

    args = parser.parse_args()
    return args


def infer_image_size(image_size):
    image_size = image_size.split(",")

    if len(image_size) > 2:
        image_size = image_size[:2]

    if len(image_size) == 1:
        image_size.append(image_size[0])

    image_size = list(map(int, image_size))
    return image_size


if __name__ == "__main__":
    args = parse_args()

    test_dataset = TestDataset(root=args.root, image_size=infer_image_size(args.image_size))
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False
    )

    model = RetinaNet.from_checkpoint(checkpoint_path=args.weights, name=args.name)
    logger = pl.loggers.TensorBoardLogger(args.output_dir, name="predictions")
    trainer = pl.Trainer(gpus=args.gpus, logger=logger)

    trainer.test(model, test_dataloaders=test_dataloader)
