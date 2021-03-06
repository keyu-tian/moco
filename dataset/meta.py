from typing import NamedTuple

import torchvision.datasets as tv_ds
from torch.utils.data import Dataset


class DatasetInfo(NamedTuple):
    train_val_set_size: int
    test_set_size: int
    img_ch: int
    img_hw: int
    num_classes: int
    clz: Dataset.__class__
    mean: tuple
    std: tuple


dataset_info = dict(
    imagenet=DatasetInfo(1281168, 50000, 3, 224, 1000, None, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    imagenet120=DatasetInfo(153487, 6000, 3, 224, 120, None, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    cifar10=DatasetInfo(50000, 10000, 3, 32, 10, tv_ds.CIFAR10, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    cifar100=DatasetInfo(50000, 10000, 3, 32, 100, tv_ds.CIFAR100, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
)




