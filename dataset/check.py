import os
import random
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def _local_check(root=None):
    if root is None:
        root = os.path.abspath(os.path.join(os.path.expanduser('~'), 'datasets', 'cifar10'))
    trs = torchvision.datasets.CIFAR10(
        root=root, train=True, download=False,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    trld = DataLoader(trs, batch_size=256, shuffle=True)
    sum_pixels, num_imgs = torch.tensor(0.), 0

    for inp, _ in trld:
        sum_pixels += inp.sum()
        num_imgs += inp.shape[0]
    
    assert sum_pixels.allclose(torch.tensor(72708632.)) and num_imgs == 50000


def _ddp_check(root=None):
    ...


if __name__ == '__main__':
    _local_check()
