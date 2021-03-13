import json
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
from math import pi

from utils.misc import set_seed

tc = torch
import torchvision

tv = torchvision
from torchvision.transforms import transforms
import torch.nn.functional as F

cols = 6


def show_ts_ims(ts_imgs):
    l = len(ts_imgs)
    global cols
    rows = (l + cols - 1) // cols
    plt.figure(figsize=(cols * 1.3, rows * 1.3))
    for i in range(l):
        plt.subplot(rows, cols, i + 1)
        im = ts_imgs[i].detach().clone().mul(STD_TS[0]).add(MEAN_TS[0])
        # im = torch.max(torch.min(im, torch.ones_like(im)), torch.zeros_like(im))
        plt.imshow(transforms.ToPILImage()(im).convert('RGB'))
    plt.show()


if __name__ == '__main__':
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    root = r'C:\Users\16333\datasets\cifar10'
    MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    MEAN_TS, STD_TS = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    
    from aug_op.ops import (
        Color,  # 饱和度
        Posterize,  # 看不出变化
        Solarize,  # 幅值最大就变成Invert，幅值不是最大反而会出噪点，建议去了
        Contrast,  # 对比度。Brightness0.9肯定可以去了。
        Brightness,  # 亮度。Brightness0.9肯定可以去了。0.6可能可以去
        Sharpness,  # 锐化
        AutoContrast,  # ？
        Equalize,  # 非常奇怪的颜色变换，还有噪点（但是噪点没有Solarize那么多，而且比较平滑）
        Invert,  # 颜色翻转,
        RandomPerspective
    )

    raw_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    aug1_trans = transforms.Compose([
        # transforms.RandomCrop(32, padding=6, padding_mode='edge'),
        
        # Color(Color.RANGES[6]),
        # Contrast(Contrast.RANGES[6]),
        # Brightness(Brightness.RANGES[5]),
        # transforms.RandomApply([Equalize()], 0.5),
        # transforms.Compose([transforms.RandomApply([AutoContrast()], 0.5), Sharpness(Sharpness.RANGES[7])]),
        # Color(Color.RANGES[7]),
    
        transforms.Compose([Color(Color.RANGES[8]), transforms.RandomApply([Brightness(Brightness.RANGES[4])], p=0.4)]),
        
        transforms.ToTensor(),
        # RandomPerspective(RandomPerspective.RANGES[4]),
        transforms.Normalize(MEAN, STD)
    ])
    aug2_trans = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    raw_tr = torchvision.datasets.CIFAR10(root=root, train=False, transform=raw_trans)
    aug1_tr = torchvision.datasets.CIFAR10(root=root, train=False, transform=aug1_trans)
    aug2_tr = torchvision.datasets.CIFAR10(root=root, train=False, transform=aug2_trans)
    
    from torch.utils.data import DataLoader
    
    set_seed(233)
    raw_ld = DataLoader(dataset=raw_tr, batch_size=cols, shuffle=False, pin_memory=True, num_workers=2)
    aug1_ld = DataLoader(dataset=aug1_tr, batch_size=cols, shuffle=False, pin_memory=True, num_workers=2)
    aug1_ld_ = DataLoader(dataset=aug1_tr, batch_size=cols, shuffle=False, pin_memory=True, num_workers=2)
    aug2_ld = DataLoader(dataset=aug2_tr, batch_size=cols, shuffle=False, pin_memory=True, num_workers=2)
    aug2_ld_ = DataLoader(dataset=aug2_tr, batch_size=cols, shuffle=False, pin_memory=True, num_workers=2)
    set_seed(233)
    
    for (raw, _), (aug1, _), (aug1_, _), (aug2, _), (aug2_, _) in zip(raw_ld, aug1_ld, aug1_ld_, aug2_ld, aug2_ld_):
        aug1_tr.transform = raw_trans
        show_ts_ims(torch.cat((raw[:cols], aug1[:cols], aug1_[:cols], aug2[:cols], aug2_[:cols])))
        # show_ts_ims(torch.cat((raw[:cols], aug1[:cols], aug2[:cols], aug2_[:cols])))
