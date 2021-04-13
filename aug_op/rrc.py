import random
import random
from typing import Tuple

import numba
import numpy as np
from torchvision.transforms import transforms


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


@numba.jit(nopython=True, nogil=True, fastmath=True)    # , parallel=True
def get_rrc_param(HW: int):
    width, height = HW, HW
    area = height * width
    
    for _ in range(10):
        target_area = random.uniform(0.08, 1.0) * area
        aspect_ratio = np.exp(random.uniform(-0.28768207245178085, 0.28768207245178085))
        w = int(round(np.sqrt(target_area * aspect_ratio)))
        h = int(round(np.sqrt(target_area / aspect_ratio)))
        if 0 < w <= width and 0 < h <= height:
            y0 = random.randint(0, height - h)
            x0 = random.randint(0, width - w)
            y1 = y0+h-1
            x1 = x0+w-1
            return y0, x0, y1, x1
        # target_area = np.random.uniform(0.08, 1.0) * area
        # aspect_ratio = np.exp(np.random.uniform(-0.28768207245178085, 0.28768207245178085))
        # w = int(round(np.sqrt(target_area * aspect_ratio)))
        # h = int(round(np.sqrt(target_area / aspect_ratio)))
        # if 0 < w <= width and 0 < h <= height:
        #     i = np.random.randint(0, height - h + 1)
        #     j = np.random.randint(0, width - w + 1)
        #     return i, j, h, w

    w = width
    h = height
    y0 = (height - h) // 2
    x0 = (width - w) // 2
    y1 = y0+h-1
    x1 = x0+w-1
    return y0, x0, y1, x1


C_in = -1
@numba.jit(nopython=True, nogil=True, fastmath=True)    # , parallel=True
def filter_param_pair(special: bool, AB_range_lb: float, AB_range_ub: float, C_range_lb: float, C_range_ub: float, HW: int, target_num: int, verbose: bool):
    params = []
    IoUs = []
    
    Cy0, Cx0, Cy1, Cx1 = HW // 4 + C_in, HW // 4 + C_in, 3 * HW // 4 - C_in, 3 * HW // 4 - C_in
    SC = ((Cy1 - Cy0 + 1) // 2) * ((Cx1 - Cx0 + 1) // 2)
    log_freq = max(1, target_num // 5)
    
    cnt = 0
    while cnt < target_num:
        Ay0, Ax0, Ay1, Ax1 = get_rrc_param(HW)
        By0, Bx0, By1, Bx1 = get_rrc_param(HW)

        
        AB = __IoU(Ay0, Ax0, Ay1, Ax1, By0, Bx0, By1, Bx1)
        AC = __IoU(Ay0, Ax0, Ay1, Ax1, Cy0, Cx0, Cy1, Cx1)
        BC = __IoU(By0, Bx0, By1, Bx1, Cy0, Cx0, Cy1, Cx1)

        if special:
            SA = (Ay1 - Ay0 + 1) * (Ax1 - Ax0 + 1)
            SB = (By1 - By0 + 1) * (Bx1 - Bx0 + 1)
            rS = min(SA / SB, SB / SA)
            ok = 0.25 > AB > -1 and 0.55 > AC > 0.23 and 0.55 > BC > 0.23 and rS > 0.5 and SA > SC / 2.5 and SB > SC / 2.5
        else:
            ok = AB_range_lb < AB < AB_range_ub and C_range_lb < AC < C_range_ub and C_range_lb < BC < C_range_ub
        if ok:
            if verbose and (cnt % log_freq == 0 or cnt == target_num - 1):
                pro: float = round(((cnt+1) / target_num * 100), 2)
                print(pro)
            cnt += 1
            Ai, Aj, Ah, Aw = Ay0, Ax0, Ay1-Ay0+1, Ax1-Ax0+1
            Bi, Bj, Bh, Bw = By0, Bx0, By1-By0+1, Bx1-Bx0+1
            params.append([Ai, Aj, Ah, Aw, Bi, Bj, Bh, Bw])
            IoUs.append([AB, AC, BC])
    
    return np.array(IoUs), np.array(params)


def get_params(HW, rrc_test_cfg: str, target_num, verbose):
    
    def get_range(s: str):
        lb, ub = s.split('_')
        return -1 if lb == '-' else float(lb), 2 if ub == '-' else float(ub)
    
    special = 'All' in rrc_test_cfg
    if not special:
        if 'Rand' in rrc_test_cfg or 'rand' in rrc_test_cfg:
            AB_range_lb = C_range_lb = -1
            AB_range_ub = C_range_ub = 2
        else:
            cfgs = [s.strip('_') for s in rrc_test_cfg.strip('AB_').split('C_')]
            if cfgs[0] == '':
                AB_range_lb, AB_range_ub = -1, 2
                C_range_lb, C_range_ub = get_range(cfgs[1])
            elif len(cfgs) == 1:
                AB_range_lb, AB_range_ub = get_range(cfgs[0])
                C_range_lb, C_range_ub = -1, 2
            else:
                AB_range_lb, AB_range_ub = get_range(cfgs[0])
                C_range_lb, C_range_ub = get_range(cfgs[1])
    else:
        AB_range_lb = -5
        AB_range_ub = -5
        C_range_lb = -5
        C_range_ub = -5
    
    IoUs, final_params = filter_param_pair(
        special=special,
        AB_range_lb=AB_range_lb,
        AB_range_ub=AB_range_ub,
        C_range_lb=C_range_lb,
        C_range_ub=C_range_ub,
        HW=HW,
        target_num=target_num,
        verbose=verbose
    )
    return IoUs, final_params
get_params(32, 'All0.0_0.3', 100, False)


if __name__ == '__main__':
    from aug_op.rrc_vis import vis
    vis(*get_params(32, 'AllAB_-_0.3_C_0.2_-', 10000, True))

