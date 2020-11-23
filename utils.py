"""
__author__: bishwarup307
Created: 21/11/20
"""

import os
from pathlib import Path
from typing import Any, Union

import torch


def ifnone(x: Any, y: Any):
    val = x if x is not None else y
    return val


def isfile(path: Union[str, os.PathLike]):
    return Path(path).is_file()


def ifdir(path: Union[str, os.PathLike], alt_path: Union[str, os.PathLike]):
    return path if Path(path).is_dir() else alt_path


def to_tensor(x: Any):
    try:
        return torch.tensor(x)
    except ValueError as exc:
        print("invalid type for tensor conversion")
        raise exc


def xyxy_to_ccwh(t: torch.Tensor) -> torch.Tensor:
    """
    converts bbox coordinates from `(xmin, ymin, xmax, ymax)` to `(x_center, y_center, width, height)`
    """
    assert (
        t.size(1) == 4
    ), "input tensor must be of size `(N, 4)` with format `(xmin, ymin, xmax, ymax)`"
    mins = t[..., :2]
    maxs = t[..., 2:]
    return torch.cat([(mins + maxs) / 2, maxs - mins], 1)


def ccwh_to_xyxy(t: torch.Tensor) -> torch.Tensor:
    """
    converts bbox coordinates from `(x_center, y_center, width, height)` to `(xmin, ymin, xmax, ymax)`
    """
    assert (
        t.size(1) == 4
    ), "input tensor must be of size `(N, 4)` with format `(x_center, y_center, width, height)`"
    cc = t[..., :2]
    wh = t[..., 2:]
    return torch.cat([cc - wh / 2, cc + wh / 2], 1)


if __name__ == "__main__":
    print(ifnone(None))
