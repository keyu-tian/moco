import json
import time

import matplotlib.pyplot as plt
import torch
from math import pi

# from mcmc_aug.agent import AugPolicy, MCMCAug
# from pipeline import BasicPipeline
from aug_op.aug import Augmenter
from aug_op.ops import GaussianBlur, Sharpness, RandSharpness
from utils.misc import set_seed
from aug_op.cspace import rgb_to_hsv, hsv_to_rgb

tc = torch
import numpy as np
import torchvision

tv = torchvision
from torchvision.transforms import transforms
import torch.nn.functional as F

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def denormalize(img):
    return img.mul(STD_TS[0]).add(MEAN_TS[0])


def normalize(img):
    return img.sub(MEAN_TS[0]).div(STD_TS[0])


def show_tv_grids(ts_imgs: torch.Tensor):
    grids: torch.Tensor = torchvision.utils.make_grid(ts_imgs, padding=1)
    plt.imshow(np.transpose(grids.numpy(), (1, 2, 0)), interpolation='nearest')
    plt.show()


cols = 10


def show_ts_ims(ts_imgs):
    l = len(ts_imgs)
    rows = (l + cols - 1) // cols
    plt.figure(figsize=(cols * 1.6, rows * 1.6))
    for i in range(l):
        plt.subplot(rows, cols, i + 1)
        im = denormalize(ts_imgs[i].detach().clone())
        # im = torch.max(torch.min(im, torch.ones_like(im)), torch.zeros_like(im))
        plt.imshow(transforms.ToPILImage()(im).convert('RGB'))
    plt.show()


def load_para(fname: str):
    if fname.startswith('theta_his'):
        with open(f'C:\\Users\\16333\\Desktop\\PyCharm\\mcmc_aug\\meta_ckpt\\{fname}.json', 'r') as fp:
            li = json.load(fp)
            para = torch.tensor(li)
            para = BasicPipeline.select_para(para, 0.5, 10000)
    else:
        para = torch.load(f'C:\\Users\\16333\\Desktop\\PyCharm\\mcmc_aug\\meta_ckpt\\{fname}.tar')['selected_para']
    print(f'selected para.shape: {para.shape}')
    return para


if __name__ == '__main__':
    fname = (
        
        # 'sea_tl0.002_aws_nosycabg_alr0.2_wr40_200ep_no5_eps0.005_aug_rk4_anoi2e-05_tail10000_31400.pth'
        # 'sea_tl0.002_aws_nosycabg_alr0.4_wr28_200ep_no5_eps0.005_aug_rk15_anoi0.001_tail10000_31400.pth'
        # 'sea_b1k0.4wd1e-4_tl0.002_r50_12ep_200ep_alr0.2_no2e-5_eps0.001_aug_rk0_anoi2e-05_tail14500_29000.pth'
        'sea_b1k0.4wd1e-4_tl0.002_r50_120ep_200ep_alr0.4_no1e-4_eps0.003_aug_rk0_anoi0.0001_tail14500_29000.pth'
        
    )
    
    if fname.endswith('.tar'):
        fname = fname.replace('.tar', '')
    
    stt = time.time()
    print('constructing the agent')
    
    # para = load_para(fname)
    para = None
    
    seed = 8
    
    
    set_seed(seed)
    aa = Augmenter(
        ch_means=MEAN, ch_stds=STD,
        adversarial=True,
        searching=[
            'color_aug',
            'blur_aug',
            'crop_aug',
        ],
        expansion=256,
        act_name='tanh',
        padding_mode='zeros',   # 'border', 'reflection' or 'zeros'
        rand_grayscale_p=0,
        target_norm=1.1,
        soft_target=0.3,
    )
    print('num para of AA: ', sum(p.numel() for p in aa.parameters()) / 1e6)
    
    print('agent constructed, time cost: %.2f' % (time.time() - stt))
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    root = r'C:\Users\16333\datasets\cifar10'
    MEAN_TS, STD_TS = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    org_tf = transforms.ToTensor()
    rrc_tf = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.15),
        # GaussianBlur([1., 1.]),
    
        RandSharpness(),
        
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
    
    set_seed(seed)
    org_loader = DataLoader(
        dataset=torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=org_tf),
        batch_size=2 * cols,
        shuffle=False, pin_memory=True,
    )
    set_seed(seed)
    rrc_loader = DataLoader(
        dataset=torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=rrc_tf),
        batch_size=2 * cols,
        shuffle=False, pin_memory=True,
    )
    set_seed(seed)
    aug_loader = DataLoader(
        dataset=torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=aug_tf),
        batch_size=2 * cols,
        shuffle=False, pin_memory=True,
    )
    set_seed(seed)
    
    op = torch.optim.Adam(aa.parameters(), lr=0.003)
    
    max_it = len(rrc_loader)
    org_itrt, rrc_itrt1, rrc_itrt2, aug_itrt = iter(org_loader), iter(rrc_loader), iter(rrc_loader), iter(aug_loader)
    for it in range(max_it):
        stt_t = time.time()
        rrc_inp1, rrc_tar1 = next(rrc_itrt1)
        rrc_inp2, rrc_tar2 = next(rrc_itrt2)
        org_inp, org_tar = next(org_itrt)
        rrc_t = time.time()

        aug_inp, aug_tar = next(aug_itrt)
        (concated_aug_vec, _, _, aug_dim, norm_p), (view1, view2) = aa(aug_inp, normalizing=True)
        aug_vec1, aug_vec2 = concated_aug_vec.data[:, :aug_dim], concated_aug_vec.data[:, aug_dim:]
        
        d = -((view1-view2).norm() ) / org_inp.norm()
        op.zero_grad()
        d.sum().backward()
        orig_norm = torch.nn.utils.clip_grad_norm_(aa.parameters(), 10)
        print(f'loss={d.item():.3g},  orig_norm={orig_norm:.3g}')
        op.step()
        aug_norm1, aug_norm2 = aug_vec1.norm(norm_p, dim=1).mean().item(), aug_vec2.norm(norm_p, dim=1).mean().item()
        aug_grad1, aug_grad2 = concated_aug_vec.grad[:, :aug_dim], concated_aug_vec.grad[:, aug_dim:]


        aug_t = time.time()
        print(f'it[{it}/{max_it}] nm1={aug_vec1.norm(dim=1).mean().item():.3g}, nm2={aug_vec2.norm(dim=1).mean().item():.3g}    org+rrc time: {rrc_t - stt_t:.3f}s, aug time: {aug_t - rrc_t:.3f}s')
        
        dc = (hsv_to_rgb(rgb_to_hsv((denormalize(view1.data)))) - (denormalize(view1.data)))
        print(f'it[{it}/{max_it}] dc.mean=', f'{dc.abs().sum() / dc.numel():.5f}', 'dc.max=', dc.abs().max(), f'dc>0.1={(dc.abs() > 0.1).sum().item() / dc.numel() * 100:.2f}%')
        
        show_ts_ims(torch.cat((
            normalize(org_inp[0 * cols:1 * cols]),
            normalize(rrc_inp1[0 * cols:1 * cols]),
            normalize(rrc_inp2[0 * cols:1 * cols]),
            view1[0 * cols:1 * cols],
            view2[0 * cols:1 * cols],
        )))
