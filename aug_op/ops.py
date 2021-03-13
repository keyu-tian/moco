import math
import random
from functools import reduce

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps

from .registry import Registry

aug_ops_dict = Registry()


@aug_ops_dict.register
class ShearX(object):
    RANGES = np.linspace(0, 0.3, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return img.transform(
            img.size, Image.AFFINE, (
                1, self.magnitude * random.choice([-1, 1]), 0, 0, 1, 0
            ), Image.BICUBIC, fillcolor=(128, 128, 128)
        )


@aug_ops_dict.register
class ShearY(object):
    RANGES = np.linspace(0, 0.3, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return img.transform(
            img.size, Image.AFFINE, (
                1, 0, 0, self.magnitude * random.choice([-1, 1]), 1, 0
            ), Image.BICUBIC, fillcolor=(128, 128, 128)
        )


@aug_ops_dict.register
class TranslateX(object):
    RANGES = np.linspace(0, 150 / 331, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return img.transform(
            img.size, Image.AFFINE, (
                1, 0, self.magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0
            ), fillcolor=(128, 128, 128)
        )


@aug_ops_dict.register
class TranslateY(object):
    RANGES = np.linspace(0, 150 / 331, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return img.transform(
            img.size, Image.AFFINE, (
                1, 0, 0, 0, 1, self.magnitude * img.size[1] * random.choice([-1, 1])
            ), fillcolor=(128, 128, 128)
        )


@aug_ops_dict.register
class Rotate(object):
    RANGES = np.linspace(0, 30, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return img.rotate(self.magnitude * random.choice([-1, 1]))


@aug_ops_dict.register
class Color(object):
    RANGES = np.linspace(0.0, 0.9, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return ImageEnhance.Color(img).enhance(
            1 + self.magnitude * random.choice([-1, 1])
        )


@aug_ops_dict.register
class Posterize(object):
    RANGES = np.round(np.linspace(8, 4, 10), 0).astype(np.int)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return ImageOps.posterize(img, self.magnitude)


@aug_ops_dict.register
class Solarize(object):
    RANGES = np.linspace(256, 0, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return ImageOps.solarize(img, self.magnitude)


@aug_ops_dict.register
class Contrast(object):
    RANGES = np.linspace(0.0, 0.9, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return ImageEnhance.Contrast(img).enhance(
            1 + self.magnitude * random.choice([-1, 1])
        )


@aug_ops_dict.register
class Sharpness(object):
    RANGES = np.linspace(0.0, 0.9, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return ImageEnhance.Sharpness(img).enhance(
            1 + self.magnitude * random.choice([-1, 1])
        )


@aug_ops_dict.register
class Brightness(object):
    RANGES = np.linspace(0.0, 0.9, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return ImageEnhance.Brightness(img).enhance(
            1 + self.magnitude * random.choice([-1, 1])
        )


@aug_ops_dict.register
class AutoContrast(object):
    RANGES = None
    
    def __init__(self):
        pass
    
    def __call__(self, img):
        return ImageOps.autocontrast(img)


@aug_ops_dict.register
class Equalize(object):
    RANGES = None
    
    def __init__(self):
        pass
    
    def __call__(self, img):
        return ImageOps.equalize(img)


@aug_ops_dict.register
class Invert(object):
    RANGES = None
    
    def __init__(self):
        pass
    
    def __call__(self, img):
        return ImageOps.invert(img)


@aug_ops_dict.register
class RandomPerspective(object):
    RANGES = np.linspace(0.0, 0.9, 10)
    grids_and_homo = {}
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
        self.rot, self.sc, self.sh, self.tr, self.ps = [torch.eye(3).float() for _ in range(5)]
    
    def __call__(self, img):
        C, H, W = img.shape
        _, homo = self._get_pos_and_homo(H, W)
        
        mags = [self.magnitude * t for t in random.choices([-1, 0, 1], weights=[1, 4, 1], k=9)]
        rot, sc_x, sc_y, sh_x, sh_y, tr_x, tr_y, ps_x, ps_y = mags
        
        mats = []
        if rot != 0:
            self.rot[0, 0] = math.cos(rot)
            self.rot[1, 0] = math.sin(rot)
            self.rot[1, 1] = self.rot[0, 0]
            self.rot[0, 1] = -self.rot[1, 0]
            mats.append(self.rot)
        if sc_x != 0 or sc_y != 0:
            self.sc[0, 0] = 1+sc_x
            self.sc[1, 1] = 1+sc_y
            mats.append(self.sc)
        if sh_x != 0 or sh_y != 0:
            self.sh[0, 1] = sh_x
            self.sh[1, 0] = sh_y
            mats.append(self.sh)
        if tr_x != 0 or tr_y != 0:
            self.tr[0, 2] = tr_x
            self.tr[1, 2] = tr_y
            mats.append(self.tr)
        if ps_x != 0 or ps_y != 0:
            self.ps[2, 0] = ps_x
            self.ps[2, 1] = ps_y
            mats.append(self.ps)

        le = len(mats)
        if le == 0:
            return img
        elif le == 1:
            t_mat = mats[0]
        else:
            random.shuffle(mats)
            t_mat = reduce(torch.matmul, mats)
        return apply_transform_to_batch(img.unsqueeze(0), t_mat.unsqueeze(0), homo, padding_mode='border', align_corners=False)[0]     # zeros, border

    def _get_pos_and_homo(self, H, W):
        s = (H, W)
        if s not in self.grids_and_homo.keys():
            wide_grids = F.affine_grid(
                torch.eye(2, 3).unsqueeze(dim=0),
                size=[1, 1, H, W],
                align_corners=True  # True, left-top: [-1, -1], right-bottom: [1, 1]
            )
            # (1, H, W, 2) => (1, 2, H, W)
            positions = wide_grids.permute(0, 3, 1, 2).contiguous()
        
            narrow_grids = F.affine_grid(
                torch.eye(2, 3).unsqueeze(dim=0),
                size=[1, 1, H, W],
                align_corners=False
            )
            # (1, H, W, 2) => (1, H, W, 3) => (1, 3, H, W) => (1, 3, H*W)
            homo_coords = F.pad(narrow_grids, [0, 1], mode='constant', value=1.).permute(0, 3, 1, 2).view(1, 3, H*W).contiguous()
        
            RandomPerspective.grids_and_homo[s] = (positions, homo_coords)
            return positions, homo_coords
        else:
            return RandomPerspective.grids_and_homo[s]


def apply_transform_to_batch(img_batch: torch.Tensor, trans_batch: torch.Tensor, homo_coords: torch.Tensor, padding_mode: str, align_corners: bool):
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