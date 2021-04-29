import itertools
import os
import sys
import time
import datetime
from copy import deepcopy
from pprint import pprint as pp
from pprint import pformat as pf
from typing import List, Tuple

import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from aug_op.cspace import rgb_to_hsv, hsv_to_rgb
from utils.misc import swish


def uniform_noise(*shape):
    return torch.rand(*shape) * 2 - 1


class UnSqueezeChannel(nn.Module):
    def forward(self, x):
        # x: (bs, fea_dim)
        return x.unsqueeze(1)


class SqueezeChannel(nn.Module):
    def forward(self, x):
        # x: (bs, fea_dim)
        return x.squeeze(1)


class FCBlock(nn.Module):
    def __init__(self, inp_d, oup_d, noi_d, act):
        super(FCBlock, self).__init__()
        self.noi_d = noi_d
        self.fc = nn.Linear(inp_d, oup_d)
        self.bn = nn.InstanceNorm1d(1, affine=True)
        self.act = act

    def forward(self, x):   # x: (B, inp_d-noi_d)
        B = x.shape[0]
        noise = uniform_noise(B, self.noi_d).to(x.device)
        noisy_x = torch.cat((x, noise), dim=1)
        
        feature = self.fc(noisy_x)
        feature = self.bn(feature.unsqueeze(1)).squeeze(1)
        feature = self.act(feature)
        
        return feature


class AugGenerator(nn.Module):
    def __init__(self, expansion, aug_dim, target_norm, soft_target, activate):
        assert 0 <= soft_target <= 1
        super(AugGenerator, self).__init__()
        self.target_norm, self.soft_target = target_norm, soft_target
        
        output_dim = aug_dim * 2
        dims = [2, 1, 1, 1, 1]
        output_dims = [d * expansion for d in dims]
        output_dims.append((output_dims[-1] + output_dim * 2) // 2)
        output_dims.append(output_dim * 2)
        output_dims.append(output_dim)
        
        input_dims = deepcopy(output_dims[:-1])
        output_dims = deepcopy(output_dims[1:])
        noise_dims = [0] * len(input_dims)
        noise_dims[1:5] = input_dims[1:5]
        input_dims = [i + n for i, n in zip(input_dims, noise_dims)]
        
        self.fcs = nn.ModuleList()
        for i, (inp_d, oup_d, noi_d) in enumerate(zip(input_dims, output_dims, noise_dims)):
            act = torch.tanh if i + 1 == len(input_dims) else activate
            self.fcs.append(FCBlock(inp_d, oup_d, noi_d, act))
        
        self.input0_dim = input_dims[0]
        self.initialize()
        
        # print(input_dims)
        # print(output_dims)
        # print(noise_dims)
    
    def initialize(self):
        for i, module in enumerate(self.fcs):
            name = module.__class__.__name__
            if 'Linear' in name:
                std = 0.005 if i + 1 == len(self.layers) else 0.005
                module.weight.data.normal_(mean=0.0, std=std)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, im_batch: Tensor):
        B = im_batch.shape[0]
        #   h   s   v      blur   tr_x, tr_y, area, ratio
        # concated_aug_vector = torch.tensor([[
        #     0., 0., 0.,     0.,     -0.5, 0., 0.5, -0.5,             0., 0., 0.,     0.,     0., 0., 0., 0.
        # ]]).repeat(B, 1)
        # return concated_aug_vector

        feature = uniform_noise(B, self.input0_dim).to(im_batch.device)
        for noisy_fc in self.fcs:
            noisy_fc: FCBlock
            feature = noisy_fc(feature)
        
        concated_aug_vector = feature
        norm = concated_aug_vector.norm(2, dim=1, keepdim=True)

        concated_aug_vector = concated_aug_vector / norm * self.target_norm
        if self.soft_target > 1e-5:
            concated_aug_vector *= 1-self.soft_target + self.soft_target * norm.sigmoid()
        
        return concated_aug_vector


class Augmenter(nn.Module):
    grids_and_homo = dict()
    eye3, I_filter, d_filter = ..., ..., ...
    dev = ...
    padding_mode = ...
    
    def __init__(
            self, ch_means: Tuple, ch_stds: Tuple,
            expansion,
            act_name,
            padding_mode,           # 'border', 'reflection' or 'zeros'
            rand_grayscale_p=0.2,
            target_norm=1., soft_target=0.2,
            searching: List[str] = None,
    ):
        super(Augmenter, self).__init__()

        Augmenter.padding_mode = padding_mode
        _ = torch.empty(1)
        if torch.cuda.is_available():
            _ = _.cuda()
        Augmenter.dev = _.device
        
        self.MEAN = torch.tensor(ch_means).float().view(1, 3, 1, 1).contiguous().to(Augmenter.dev)
        self.STD = torch.tensor(ch_stds).float().view(1, 3, 1, 1).contiguous().to(Augmenter.dev)
        
        ls = [
            (Augmenter.color_aug, 3),
            (Augmenter.blur_aug, 1),
            (Augmenter.crop_aug, 4),
        ]
        if searching is not None:
            ls = list(filter(lambda tu: tu[0].__name__ in searching, ls))
        self.aug_funcs, self.aug_param_lens = list(zip(*ls))
        pref_sum = list(itertools.accumulate(self.aug_param_lens))
        self.begs, self.ends = tuple([0] + pref_sum[:-1]), tuple(pref_sum)

        self.aug_dim = sum(self.aug_param_lens)
        act = {
            'tanh': torch.tanh,
            'sigmod': torch.sigmoid,
            'relu': F.relu,
            'relu6': F.relu6,
            'swish': swish,
        }[act_name]
        self.generator = AugGenerator(
            expansion=expansion,
            aug_dim=self.aug_dim,
            target_norm=target_norm, soft_target=soft_target,
            activate=act
        )
        Augmenter.eye3 = torch.eye(3).reshape(1, 3, 3).to(Augmenter.dev)
        Augmenter.I_filter = torch.tensor([[
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
        ]]).to(Augmenter.dev)
        Augmenter.d_filter = torch.tensor([[
            [-0.05, -0.10, -0.05],
            [-0.10, +0.60, -0.10],
            [-0.05, -0.10, -0.05],
        ]]).to(Augmenter.dev)
        
        self.rand_grayscale = rand_grayscale_p is not None and rand_grayscale_p > 1e-4
        self.rand_grayscale_p = rand_grayscale_p
    
    def split_aug_params(self, aug_params):
        two_augs = []
        for p in (aug_params[:, :self.aug_dim], aug_params[:, self.aug_dim:]):
            params = [p[:, beg:end] for beg, end in zip(self.begs, self.ends)]
            two_augs.append(params)
        return two_augs

    def normalize(self, img):
        return (img - self.MEAN) / self.STD

    def denormalize(self, img):
        return (img * self.STD) + self.MEAN
    
    def forward(self, im_batch: Tensor, normalizing=True):
        im_batch = torch.clamp(im_batch, min=0., max=1.)
        
        oup: Tensor = self.generator(im_batch)  # todo: 要把图片输入进去是不是会导致网络很大？要输的话是不是输6通道比较好？如果直接输入一个一维gaussian呢？
        two_aug_vectors = (oup[:, :self.aug_dim], oup[:, self.aug_dim:])
        
        two_views = []
        for aug_vector in two_aug_vectors:
            aug_imgs = im_batch
            for beg, end, func in zip(self.begs, self.ends, self.aug_funcs):
                param = aug_vector[:, beg:end]
                aug_imgs = func(param, aug_imgs)
            
            if self.rand_grayscale:
                aug_imgs = self.grayscale(self.rand_grayscale_p, aug_imgs)
            
            if normalizing:
                aug_imgs = self.normalize(aug_imgs)
            
            two_views.append(aug_imgs)
        
        return two_views[0], two_views[1]

    @staticmethod
    def color_aug(color_ps: Tensor, rgb_imgs: Tensor):
        B, C, H, W = rgb_imgs.shape
        d_h, d_s, d_v = color_ps.unbind(dim=1)  # (B, )
        d_h: Tensor = (0.15 * d_h).view(B, 1, 1, 1)
        d_s: Tensor = (0.5 * d_s).view(B, 1, 1, 1)
        d_v: Tensor = (0.5 * d_v).view(B, 1, 1, 1)
        
        # print('dh:', d_h.data.view(-1)[:10])
        # print('ds:', d_s.data.view(-1)[:10])
        # print('dv:', d_v.data.view(-1)[:10])
        
        d = torch.cat((d_h, d_s, d_v), dim=1)   # (B, 3, 1, 1)

        hsv_imgs = torch.clamp(rgb_to_hsv(rgb_imgs) + d, min=0., max=1.)
        aug_imgs = torch.clamp(hsv_to_rgb(hsv_imgs), min=0., max=1.)
        
        return aug_imgs

    @staticmethod
    def blur_aug(blur_ps: Tensor, rgb_imgs: Tensor):
        sigma = 1.4 * blur_ps
        sigma = sigma * torch.where(sigma > 0, torch.ones_like(sigma), 0.5 * torch.ones_like(sigma))
        
        # print('blur:', sigma.data.view(-1)[:10])
        
        B, C, H, W = rgb_imgs.shape
        kernel = sigma.view(B, 1, 1) * Augmenter.d_filter + Augmenter.I_filter    # (B, 1, 1) * (1, k, k) + (1, k, k) => (B, k, k)
        k = kernel.shape[-1]
        kernel = torch.stack((kernel, kernel, kernel), dim=1)       # (B, 3, k, k)
        stacked_kernel = kernel.view(B*C, 1, k, k)
        
        pad = k // 2
        #                                              todo: reflect or replicate
        stacked_img = F.pad(rgb_imgs, [pad, pad, pad, pad], mode='reflect').view(1, B*C, H+2*pad, W+2*pad)
        stacked_img = F.conv2d(stacked_img, stacked_kernel, bias=None, groups=B*C)
        filtered_img = torch.clamp(stacked_img.view(B, C, H, W), min=0., max=1.)

        return filtered_img

    @staticmethod
    def crop_aug(crop_ps: Tensor, rgb_imgs: Tensor):
        B, C, H, W = rgb_imgs.shape
        tr_x, tr_y, area, ratio = crop_ps.unbind(dim=1) # (B, )

        tr_x: Tensor = -0.7 * tr_x                        # -1: -0.7    0: 0      1: 0.7
        tr_y: Tensor = -0.7 * tr_y                        # -1: -0.7    0: 0      1: 0.7
        area: Tensor = 0.8 - 0.72 * area.abs().pow(0.5)   # -1: 0.08    0: 0.8    1: 0.08
        
        ratio: Tensor = ratio * np.log(3/2)
        ratio = ratio.exp()                     # -1: a/b     0: 1.0    1: b/a
        
        sc_x: Tensor = (area / ratio).sqrt()
        sc_y: Tensor = sc_x * ratio

        M_scale = Augmenter.eye3.repeat(B, 1, 1)   # (B, 3, 3)
        M_scale[:, 0, 0] = sc_x
        M_scale[:, 1, 1] = sc_y

        M_trans = Augmenter.eye3.repeat(B, 1, 1)   # (B, 3, 3)
        M_trans[:, 0, 2] = tr_x
        M_trans[:, 1, 2] = tr_y
        # todo: 平移出去越界怎么办，要把area控制住吗？那如果clip的硬控制住，还怎么保证还能传梯度？还是说一旦越界就让他们都收敛点？
        # todo: scale出去越界怎么办？因为当area接近1而ratio很悬殊的时候，长边就会伸出去

        trans_matrices = torch.matmul(M_trans, M_scale) # scale first, then translate
        
        homo = Augmenter._get_homo(H, W)
        rgb_imgs = Augmenter._apply_transform_to_batch(rgb_imgs, trans_matrices, homo, Augmenter.padding_mode)
        return torch.clamp(rgb_imgs, min=0., max=1.)

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
            ).to(Augmenter.dev)
            # (1, H, W, 2) => (1, H, W, 3) => (1, 3, H, W) => (1, 3, H*W)
            homo_coords = F.pad(narrow_grids, [0, 1], mode='constant', value=1.).permute(0, 3, 1, 2).view(1, 3, H*W).contiguous()
            Augmenter.grids_and_homo[s] = homo_coords
            return homo_coords

    @staticmethod   # todo 不debug的时候记得用 reflection 而不是 zeros
    def _apply_transform_to_batch(img_batch: Tensor, trans_batch: Tensor, homo_coords: Tensor, padding_mode='border', align_corners=False):   # todo: 'border', 'reflection' or 'zeros'
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

    @staticmethod
    def grayscale(prob: float, img: Tensor):
        B = img.shape[0]
        r, g, b = img.unbind(1)
        gray: Tensor = r * 299/1000 + g * 587/1000 + b * 114/1000
        gray = gray.unsqueeze(1).repeat(1, 3, 1, 1)
        
        mask = torch.bernoulli(torch.empty(B, 1, 1, 1), p=prob).to(img.device)
        # print(f'mask ratio: {mask.mean() * 100:5.2f}%')
        return mask * gray + (1-mask) * img
        

def main():
    Augmenter(
        ch_means=(0.4914, 0.4822, 0.4465), ch_stds=(0.2023, 0.1994, 0.2010),
        searching=[
            'color_aug',
            'blur_aug',
            'crop_aug',
        ],
        expansion=128,
        act_name='swish',
        padding_mode='zeros',   # 'border', 'reflection' or 'zeros'
        rand_grayscale_p=0,
        target_norm=1.,
        soft_target=False,
    )(torch.rand(3, 4, 5, 6))


if __name__ == '__main__':
    main()
