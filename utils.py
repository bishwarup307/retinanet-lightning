"""
__author__: bishwarup.b (bishwarup.b@eagleview.com)
Created: 21/11/20
Copyright EagleView, 2020
"""
from typing import Any

import torch


def ifnone(x: Any, y: Any):
    val = x if x is not None else y
    return val


def to_tensor(x: Any):
    try:
        return torch.tensor(x)
    except ValueError as exc:
        print("invalid type for tensor conversion")
        raise exc


if __name__ == "__main__":
    print(ifnone(None))
