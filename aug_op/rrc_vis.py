import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from utils.misc import set_seed

tc = torch
import numpy as np
import torchvision

tv = torchvision
from torchvision.transforms import transforms

root = r'C:\Users\16333\datasets\cifar10'
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
MEAN_TS, STD_TS = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
cols = 10


def denormalize(img):
    return img.mul(STD_TS[0]).add(MEAN_TS[0])


def normalize(img):
    return img.sub(MEAN_TS[0]).div(STD_TS[0])


def show_ts_ims(ts_imgs, IoUs, final_params):
    HW = 32

    from aug_op.rrc import C_in
    Ci = Cj = HW // 4 + C_in
    Cw = Ch = HW // 2 - 2 * C_in
    
    l = len(ts_imgs)
    rows = (l + cols - 1) // cols
    plt.figure(figsize=(cols * 1.8, rows * 2.2))
    plt.tight_layout(pad=15)
    for i in range(l):
        AB, AC, BC = IoUs[i]
        Ai, Aj, Ah, Aw, Bi, Bj, Bh, Bw = final_params[i]
        ax = plt.subplot(rows, cols, i + 1)
        plt.title(f'{AB:.2f}({AC:.2f}, {BC:.2f})')
        ax.add_patch(patches.Rectangle((Aj, Ai), Aw, Ah, linewidth=1, edgecolor='r', facecolor='none'))
        ax.add_patch(patches.Rectangle((Bj, Bi), Bw, Bh, linewidth=1, edgecolor='b', facecolor='none'))
        ax.add_patch(patches.Rectangle((Cj, Ci), Cw, Ch, linewidth=1, edgecolor='black', facecolor='none'))
        im = denormalize(ts_imgs[i].detach().clone())
        # im = torch.max(torch.min(im, torch.ones_like(im)), torch.zeros_like(im))
        plt.imshow(transforms.ToPILImage()(im).convert('RGB'))
    plt.show()


def vis(IoUs, final_params):
    raw_tf = transforms.Compose([
        # transforms.RandomCrop(32, padding=4, padding_mode='edge'),  # edge, constant
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=MEAN, std=STD),
    ])
    aug_tf = transforms.Compose([
        # transforms.RandomCrop(32, padding=4, padding_mode='constant'),  # edge, constant
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    from torch.utils.data import DataLoader
    
    set_seed(1)
    raw_loader = DataLoader(
        dataset=torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=raw_tf),
        batch_size=4 * cols, num_workers=2, shuffle=False, pin_memory=True,
    )
    set_seed(1)
    aug_loader = DataLoader(
        dataset=torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=aug_tf),
        batch_size=4 * cols, num_workers=2, shuffle=False, pin_memory=True,
    )
    set_seed(1)
    
    max_it = len(raw_loader)
    raw_itrt, aug_itrt = iter(raw_loader), iter(aug_loader)
    for it in range(max_it):
        raw_inp, raw_tar = next(raw_itrt)
        show_ts_ims(torch.cat((
            normalize(raw_inp[0 * cols:1 * cols]),
            normalize(raw_inp[1 * cols:2 * cols]),
            normalize(raw_inp[2 * cols:3 * cols]),
            # normalize(raw_inp[3 * cols:4 * cols]),
        )), IoUs[it*(cols*3) : (it+1)*(cols*3)], final_params[it*(cols*3) : (it+1)*(cols*3)])

