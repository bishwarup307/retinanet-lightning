# RetinaNet (pytorch-lightning)
RetinaNet is a one-stage object detection architecture introduced in 2017. Please refer to the [original paper](https://arxiv.org/abs/1708.02002) for more details.

## Training

Prerequisites:

- CUDA enabled GPU
- CUDA version 10.1 or higher

1. Clone the repo:

```sh
git clone https://github.com/bishwarup307/retinanet-lightning.git
```

2. install the requirements (preferably in a virtual environment):

```sh
cd retinanet-lightning
pip install -r requirements.txt
```

3. Format your dataset:

Your data needs to be in COCO format and have the following directory structure:

```
root
    ├── images
    │    ├── train
    │    ├── val
    │    ├── test
    │
    └── annotations
         ├── train.json
         ├── val.json
         ├── test.json
```

4. make changes in the `retinanet/config.yaml`. All your training parameters (e.g., learning rate, batch size, augmentations and more) can be configured there. 

5. Run training
```sh
python train.py
```

6. Run tensorboard:
```sh
tensborboard --logdir <logdir>/lightning_logs
```

where `logdir` is the logdir you specified in the `config.yaml`.


## Distributed Training

If you want to run the training in distributed mode just specify the number of gpus (`gpus`) you want to use in the `Trainer` section of `config.yaml`. By default it uses torch DistributedDataParallel mode. See [pytorch lightning multi-gpu training](https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html?highlight=distributed) for more options.

## FP-16 Training

If you want to take advantage of 16 bit precision training, you can also do that by setting `amp` as `True` in the `Trainer` section of `config.yaml`. We recommend using `native` as your `amp_backend` which uses [pytorch's native automatic mixed precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html). However, in case you want to use [APEX](https://github.com/NVIDIA/apex), that can be configured as well.


## Testing

```sh
python test.py \
--root /directory/with/test/images \
--image-size 512 \
--weights path/to/checkpoint.ckpt \
--batch-size 8 \
--output-dir path/to/output/dir \
```

## Export to ONNX

The best model weights (best epoch) is exported to ONNX named `best.onnx` inside the logdir as part of the training procedure.  

The ONNX model outputs `anchors`, `logits` and `offsets` where:
- `anchors` are the anchor boxes. Shape: `(A, 4)`, where `A` is the number of anchors given an image
- `logits` are the class confidences Shape: `(A, C)` where C is the number of classes
- `offsets` are the normalized offsets to the `anchors`. Shape `(A, 4)`.
