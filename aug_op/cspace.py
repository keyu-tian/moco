"""
References:
    - https://github.com/kornia/kornia
    - https://en.wikipedia.org/wiki/HSL_and_HSV
"""

import torch


def _get_sl_sv(maxc: torch.Tensor, minc: torch.Tensor, c_range: torch.Tensor, eps: float, to_sl: bool):
    if to_sl:
        l: torch.Tensor = (maxc + minc).div_(2)  # lightness, (B, H, W)
        dvs = 1 - (l + l - 1).abs()
        s: torch.Tensor = torch.where(dvs.abs() < eps, torch.zeros_like(dvs), c_range / dvs)
        return s, l
    else:
        v: torch.Tensor = maxc  # brightness, (B, H, W)
        s: torch.Tensor = torch.where(v < eps, torch.zeros_like(v), c_range / v)
        return s, v


def _rgb_to_hsl_hsv(image: torch.Tensor, eps: float, to_hsl=True) -> torch.Tensor:
    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]
    
    maxc: torch.Tensor = image.max(-3)[0]  # (B, H, W)
    minc: torch.Tensor = image.min(-3)[0]  # (B, H, W)
    c_range: torch.Tensor = maxc - minc
    
    # avoid division by zero
    c_range_is_zero = c_range < eps
    c_range_non_zero = torch.where(c_range_is_zero, torch.ones_like(c_range), c_range)
    
    rc: torch.Tensor = r / c_range_non_zero
    gc: torch.Tensor = g / c_range_non_zero
    bc: torch.Tensor = b / c_range_non_zero
    
    maxg: torch.Tensor = g == maxc
    maxr: torch.Tensor = r == maxc
    
    h: torch.Tensor = 4.0 + rc - gc
    h[maxg] = 2.0 + bc[maxg] - rc[maxg]
    h[maxr] = (gc[maxr] - bc[maxr]) % 6
    h[c_range_is_zero] = 0.0
    
    h /= 6
    
    # h = 2 * pi * h
    return torch.stack((h, *_get_sl_sv(maxc, minc, c_range, eps * 3, to_sl=to_hsl)), dim=-3)


def rgb_to_hsl(image: torch.Tensor, eps=3 / 256):
    return _rgb_to_hsl_hsv(image, eps, to_hsl=True)


def rgb_to_hsv(image: torch.Tensor, eps=3 / 256):
    return _rgb_to_hsl_hsv(image, eps, to_hsl=False)


def hsl_to_rgb(image: torch.Tensor) -> torch.Tensor:
    # h: torch.Tensor = image[..., 0, :, :]  # / (2 * pi)
    # s: torch.Tensor = image[..., 1, :, :]
    # l: torch.Tensor = image[..., 2, :, :]
    #
    # c = (1 - (l + l - 1).abs()) * s
    # hp = h * 6
    # x = c * (1 - (hp % 2 - 1))
    # z = torch.zeros_like(x)
    #
    # hi = torch.floor(hp).long() % 6
    # hi_us = hi.unsqueeze(dim=-3)
    # h01, h12, h23, h34, h45, h56 = (
    #     (hi_us == 0).expand_as(image),
    #     (hi_us == 1).expand_as(image),
    #     (hi_us == 2).expand_as(image),
    #     (hi_us == 3).expand_as(image),
    #     (hi_us == 4).expand_as(image),
    #     (hi_us == 5).expand_as(image),
    # )
    # out: torch.Tensor = torch.stack((hi, hi, hi), dim=-3).float()
    # out[h01] = torch.stack((c, x, z), dim=-3)[h01]
    # out[h12] = torch.stack((x, c, z), dim=-3)[h12]
    # out[h23] = torch.stack((z, c, x), dim=-3)[h23]
    # out[h34] = torch.stack((z, x, c), dim=-3)[h34]
    # out[h45] = torch.stack((x, z, c), dim=-3)[h45]
    # out[h56] = torch.stack((c, z, x), dim=-3)[h56]
    #
    # out += (l - c / 2).unsqueeze(dim=-3)
    #
    # return out

    h: torch.Tensor = image[..., 0, :, :]  # / (2 * pi)
    s: torch.Tensor = image[..., 1, :, :]
    l: torch.Tensor = image[..., 2, :, :]
    a = s * torch.min(l, 1-l)
    
    def f(n):
        k: torch.Tensor = n + (h * 12) % 12
        return l - a * torch.max(-torch.ones_like(k), torch.min(torch.min(k-3, 9-k), torch.ones_like(k)))
    
    return torch.stack((f(0), f(8), f(4)), dim=-3)


def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    h: torch.Tensor = image[..., 0, :, :]  # / (2 * pi)
    s: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]
    
    hi: torch.Tensor = torch.floor(h * 6).long() % 6
    f: torch.Tensor = ((h * 6) % 6) - hi
    one: torch.Tensor = torch.tensor(1.).to(image.device)
    p: torch.Tensor = v * (one - s)
    q: torch.Tensor = v * (one - f * s)
    t: torch.Tensor = v * (one - (one - f) * s)
    
    hi_us = hi.unsqueeze(dim=-3)
    h01, h12, h23, h34, h45, h56 = (
        (hi_us == 0).expand_as(image),
        (hi_us == 1).expand_as(image),
        (hi_us == 2).expand_as(image),
        (hi_us == 3).expand_as(image),
        (hi_us == 4).expand_as(image),
        (hi_us == 5).expand_as(image),
    )
    out: torch.Tensor = torch.stack((hi, hi, hi), dim=-3).float()
    
    out = out.float()
    out[h01] = torch.stack((v, t, p), dim=-3)[h01]
    out[h12] = torch.stack((q, v, p), dim=-3)[h12]
    out[h23] = torch.stack((p, v, t), dim=-3)[h23]
    out[h34] = torch.stack((p, q, v), dim=-3)[h34]
    out[h45] = torch.stack((t, p, v), dim=-3)[h45]
    out[h56] = torch.stack((v, p, q), dim=-3)[h56]
    
    return out
    # h: torch.Tensor = image[..., 0, :, :]  # / (2 * pi)
    # s: torch.Tensor = image[..., 1, :, :]
    # v: torch.Tensor = image[..., 2, :, :]
    #
    # def f(n):
    #     k: torch.Tensor = (n + h * 6) % 6
    #     return v - v * s * torch.max(torch.zeros_like(k), torch.min(torch.min(k, 4-k), torch.ones_like(k)))
    #
    # r, g, b = f(5), f(3), f(1)
    # return torch.stack((r, g, b), dim=-3)
