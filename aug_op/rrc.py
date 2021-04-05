import itertools
import json
import math
import random
import numba
import numpy as np

import torch
import tqdm
from PIL import Image
from easydict import EasyDict
from torchvision.transforms import functional as F


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """
    
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
    
    def get_params(self, img):
        width, height = img.size
        area = height * width
    
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
        
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
        
            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w
    
        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(self.ratio)):
            w = width
            h = int(round(w / min(self.ratio)))
        elif (in_ratio > max(self.ratio)):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img, i, j, h, w):
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)


@numba.jit(nopython=True, nogil=True, fastmath=True)    # , parallel=True
def __IoU(Ay0, Ax0, Ay1, Ax1, By0, Bx0, By1, Bx1):
    top_left_y = max(Ay0, By0)
    top_left_x = max(Ax0, Bx0)
    btm_right_y = min(Ay1, By1)
    btm_right_x = min(Ax1, Bx1)
    
    SA = (Ay1 - Ay0 + 1) * (Ax1 - Ax0 + 1)
    SB = (By1 - By0 + 1) * (Bx1 - Bx0 + 1)
    Sinter = max(0, btm_right_y - top_left_y + 1) * max(0, btm_right_x - top_left_x + 1)
    
    return Sinter / (SA + SB - Sinter)


C_in = -1
@numba.jit(nopython=True, nogil=True, fastmath=True)    # , parallel=True
def filter_param_pair(HW: int, sample_ratio: float, coords: np.ndarray):
    params = []
    IoUs = []
    n: int = coords.shape[0]
    
    Cy0, Cx0, Cy1, Cx1 = HW // 4 + C_in, HW // 4 + C_in, 3 * HW // 4 - C_in, 3 * HW // 4 - C_in
    SC = ((Cy1 - Cy0 + 1) // 2) * ((Cx1 - Cx0 + 1) // 2)
    
    for i in range(n):
        if i % 7000 == 0 or i == n-1:
            pro: float = round(((i+1) / n * 100), 2)
            print(pro)
        for j in range(i, n):
            # todo: remove
            if random.random() > sample_ratio:
                continue
                
            Ay0, Ax0, Ay1, Ax1 = coords[i]
            By0, Bx0, By1, Bx1 = coords[j]

            SA = (Ay1 - Ay0 + 1) * (Ax1 - Ax0 + 1)
            SB = (By1 - By0 + 1) * (Bx1 - Bx0 + 1)
            rS = min(SA / SB, SB / SA)
            
            AB = __IoU(Ay0, Ax0, Ay1, Ax1, By0, Bx0, By1, Bx1)
            AC = __IoU(Ay0, Ax0, Ay1, Ax1, Cy0, Cx0, Cy1, Cx1)
            BC = __IoU(By0, Bx0, By1, Bx1, Cy0, Cx0, Cy1, Cx1)

            # if random.random() < 0.1:
            if AC > 0.2 and BC > 0.2:
            # if 0.25 > AB > -1 and 0.55 > AC > 0.23 and 0.55 > BC > 0.23 and rS > 0.5 and SA > SC / 2.5 and SB > SC / 2.5:
                Ai, Aj, Ah, Aw = Ay0, Ax0, Ay1-Ay0+1, Ax1-Ax0+1
                Bi, Bj, Bh, Bw = By0, Bx0, By1-By0+1, Bx1-Bx0+1
                params.append([Ai, Aj, Ah, Aw, Bi, Bj, Bh, Bw])
                IoUs.append([AB, AC, BC])
    
    return np.array(IoUs), np.array(params)
# filter_param_pair(32, 0.05, np.zeros((100, 4)))
# filter_param_pair(32, 0.05, np.zeros((100, 4)))
# filter_param_pair(32, 0.05, np.zeros((100, 4)))

            
def __test():
    import os

    HW = 32
    
    def check_coord(coord):
        y0, x0, y1, x1 = coord
        if y1 <= y0 or x1 <= x0:
            return False
        #  param            coord
        # 0 1 2 3    0    1     2      3
        i, j, h, w = y0, x0, y1-y0+1, x1-x0+1
        if h * w < (HW * HW * 0.08 * 0.975):
            return False
        ratio = min(h / w, w / h)
        if ratio < 3. / 4. * 0.975:
            return False
        return True

    r = RandomResizedCrop(HW)
    N = 50000
    sum_IoU = 0.
    for _ in range(N):
        Ai, Aj, Ah, Aw = r.get_params(EasyDict({'size': (HW, HW)}))
        Ay0, Ax0, Ay1, Ax1 = Ai, Aj, Ai+Ah-1, Aj+Aw-1

        Bi, Bj, Bh, Bw = r.get_params(EasyDict({'size': (HW, HW)}))
        By0, Bx0, By1, Bx1 = Bi, Bj, Bi+Bh-1, Bj+Bw-1

        sum_IoU += __IoU(Ay0, Ax0, Ay1, Ax1, By0, Bx0, By1, Bx1)
    print(f'mean IoU = {sum_IoU/N:.3f}')
    
    params_f = f'rrc_{HW}x{HW}_params.npy'
    if not os.path.isfile(params_f):
        params = []
        for y0, x0, y1, x1 in filter(check_coord, itertools.product(*[range(HW)] * 4)):
            i, j, h, w = y0, x0, y1-y0+1, x1-x0+1
            params.append((i, j, h, w))
        
        assert len(params) == len(set(params))
        params = np.array(params)
        np.save(params_f.replace('.npy', ''), params)
    else:
        with open(params_f, 'r') as fin:
            params = np.load(params_f)
        print(f'params.shape={params.shape}')       # HW=32: (42556, 4)
    
    coords = np.stack((
        params[:, 0],                       # y0 = i
        params[:, 1],                       # x0 = j
        params[:, 0] + params[:, 2] - 1,    # y1 = i+h-1
        params[:, 1] + params[:, 3] - 1,    # x1 = j+w-1
    ), 1)
    
    final_params = np.stack((
        coords[:, 0],                       # i = y0
        coords[:, 1],                       # j = x0
        coords[:, 2] - coords[:, 0] + 1,    # h = y1-y0+1
        coords[:, 3] - coords[:, 1] + 1,    # w = x1-x0+1
    ), 1)
    
    assert np.allclose(final_params, params)

    sample_ratio = 0.003
    IoUs, final_params = filter_param_pair(HW, sample_ratio, coords)
    print(f'selective ratio = {100 * IoUs.shape[0] / (params.shape[0] ** 2 / 2 * sample_ratio):.2f}%')
    
    ML = 300000
    if len(IoUs) > ML:
        IoUs, final_params = IoUs[:ML], final_params[:ML]
        print(f'(clipped to {ML})')
    idx = np.random.permutation(len(IoUs))
    IoUs, final_params = IoUs[idx], final_params[idx]
    print(f'IoUs.shape = {IoUs.shape}')
    print(f'final_params.shape = {final_params.shape}')
    
    np.save('unnamed', final_params)
    
    from aug_op.rrc_vis import vis
    vis(IoUs, final_params)
    
    


if __name__ == '__main__':
    __test()
    