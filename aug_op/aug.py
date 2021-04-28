import itertools
import os
import sys
import time
import datetime
from pprint import pprint as pp
from pprint import pformat as pf
from typing import List, Tuple

import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from aug_op.rgb import rgb_to_hsv, hsv_to_rgb


class AugGenerator(nn.Module):
    def __init__(self, aug_dim):
        super(AugGenerator, self).__init__()
        output_dim = aug_dim * 2
        
    def forward(self, im_batch: Tensor):
        return torch.empty(2)


class Augmenter(nn.Module):
    grids_and_homo = dict()
    eye3 = ...
    
    def __init__(self, searching: List[str]):
        super(Augmenter, self).__init__()
        ls = [
            (Augmenter.color_aug, 3),
            (Augmenter.blur_aug, 1),
            (Augmenter.crop_aug, 4),
        ]
        ls = list(filter(lambda tu: tu[0].__name__ in searching, ls))
        self.aug_funcs, self.aug_param_lens = list(zip(*ls))
        pref_sum = list(itertools.accumulate(self.aug_param_lens))
        self.begs, self.ends = tuple([0] + pref_sum[:-1]), tuple(pref_sum)

        self.aug_dim = sum(self.aug_param_lens)
        self.generator = AugGenerator(self.aug_dim)
        Augmenter.eye3 = torch.eye(3).reshape(1, 3, 3).cuda()
    
    def split_aug_params(self, aug_params):
        two_augs = []
        for p in (aug_params[:, :self.aug_dim], aug_params[:, self.aug_dim:]):
            params = [p[:, beg:end] for beg, end in zip(self.begs, self.ends)]
            two_augs.append(params)
        return two_augs
    
    def forward(self, im_batch: Tensor):
        im_batch = torch.clamp(im_batch, min=0., max=1.)
        hsv_imgs = rgb_to_hsv(im_batch)
        
        oup: Tensor = self.generator(hsv_imgs)  # todo: 要把图片输入进去是不是会导致网络很大？如果直接输入一个一维gaussian呢？
        two_aug_vectors = (oup[:, :self.aug_dim], oup[:, self.aug_dim:])
        
        two_views = []
        for aug_vector in two_aug_vectors:
            aug_imgs = hsv_imgs
            for beg, end, func in zip(self.begs, self.ends, self.aug_funcs):
                param = aug_vector[:, beg:end]
                aug_imgs = torch.clamp(func(param, aug_imgs), min=0., max=1.)
            
            rgb_imgs = hsv_to_rgb(aug_imgs)
            two_views.append(rgb_imgs)
        
        return two_views

    @staticmethod
    def color_aug(color_ps: Tensor, imgs: Tensor):
        return imgs

    @staticmethod
    def blur_aug(blur_ps: Tensor, imgs: Tensor):
        return imgs

    @staticmethod
    def crop_aug(crop_ps: Tensor, imgs: Tensor):
        B, C, H, W = imgs.shape
        tr_x, tr_y, area, ratio = crop_ps.unbind(dim=1)
        
        tr_x = tr_x * 0.5
        tr_y = tr_y * 0.5
        area = area.abs() * 0.2 # todo 最小值取多少？取决于rrcmin几最好；默认是0.2
        
        

        width = (area / ratio).sqrt()
        height = width * ratio

        M_trans = Augmenter.eye3.repeat(B, 1, 1)   # (B, 3, 3)
        M_trans[:, 0, 2] = tr_x
        M_trans[:, 1, 2] = tr_y
        
        _, homo = Augmenter._get_homo(H, W)
        imgs = Augmenter._apply_transform_to_batch(imgs, trans_matrices, homo)
        return imgs

    @staticmethod
    def _get_homo(H, W):
        s = (H, W)
        if s in Augmenter.grids_and_homo.keys():
            return Augmenter.grids_and_homo[s]
        else:
            narrow_grids = F.affine_grid(
                torch.eye(2, 3).unsqueeze(dim=0),
                size=[1, 1, H, W],
                align_corners=False
            ).cuda()
            # (1, H, W, 2) => (1, H, W, 3) => (1, 3, H, W) => (1, 3, H*W)
            homo_coords = F.pad(narrow_grids, [0, 1], mode='constant', value=1.).permute(0, 3, 1, 2).view(1, 3, H*W).contiguous()
            Augmenter.grids_and_homo[s] = homo_coords
            return homo_coords

    @staticmethod
    def _apply_transform_to_batch(img_batch: torch.Tensor, trans_batch: torch.Tensor, homo_coords: torch.Tensor, padding_mode='reflection', align_corners=False):   # todo: 'border' or 'reflection'
        """
        :param img_batch: (B, C, H, W)
        :param trans_batch: (B, 3, 3)
        :param homo_coords: (1, 3, H*W)
        :param padding_mode:
        :param align_corners:
        :return: (B, C, H, W)
        """
        B, _, H, W = img_batch.shape
        t_homo_coords = trans_batch.matmul(homo_coords)   # (B, 3, 3) @ (B (1=broadcast=>B), 3, H*W) => (B, 3, H*W)
    
        t_homo_coords = t_homo_coords.view(B, 3, H, W).permute(0, 2, 3, 1)  # (B, 3, H*W) => (B, 3, H, W) => (B, H, W, 3)
        w = t_homo_coords[:, :, :, -1:]
        ones = torch.ones_like(w)
        w = torch.where(w > 1e-6, w, ones)
        cartesian_coords = t_homo_coords[:, :, :, :-1] / w               # (B, H, W, 3) => (B, H, W, 2)
    
        return F.grid_sample(img_batch, cartesian_coords, mode='bilinear', padding_mode=padding_mode, align_corners=align_corners)


def main():
    print(Augmenter([
        'color_aug',
        'blur_aug',
        'crop_aug',
    ]).aug_param_lens)


if __name__ == '__main__':
    main()
