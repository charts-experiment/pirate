"""From 8 bits (.fil, floor 32, σ = 4 counts) to the 4 bits + scale/offset that pirate receives.

Input:  block [T, nchan] uint8 in the .fil channel order (descending frequency).
Output: codes [nchan, T] uint8 (0..15 = two's-complement int4) with channel 0 = lowest frequency,
        and scales/offsets [nchan, T/256, 2] float16 per minichunk.

Two modes (the same as the original emitter of the earlier chain):
  fixed        q = floor((x − 32)/2) clipped to [−7, 7]; scale 0.5 σ, offset 0.125 σ.
  per_channel  median and σ = (p84 − p16)/2 of each channel in the block (using 1 in 4 samples);
               q = rint((x − median)/(σ/2)) in [−7, 7]; scale 0.5, offset 0; channels with σ < 1 or median > 250
               and samples at 255 are sent as missing (−8). This is what the search node's converter would do.
Optional mask: flagged channels = −8 and the scale of the good ones × √(N/N_good), because pirate computes the
S/N assuming unit variance in every channel.

Works with numpy or cupy (`xp`).
"""
import numpy as np

_LUT = np.clip(np.floor((np.arange(256) - 32) / 2.0), -7, 7).astype(np.int8)
FIXED_LUT = (_LUT & 0x0F).astype(np.uint8)
MISSING = 8                                       # code of −8


def base_scales(nchan, T, per_channel, xp=np):
    so = xp.zeros((nchan, T // 256, 2), dtype=xp.float16)
    so[..., 0] = 0.5
    so[..., 1] = 0.0 if per_channel else 0.125
    return so


def apply_mask(codes, so, mask_asc, xp=np):
    """mask_asc: bool [nchan] in ascending frequency order. Modifies codes and so in place."""
    if mask_asc is None:
        return
    m = xp.asarray(mask_asc)
    codes[m, :] = MISSING
    corr = float(np.sqrt(m.size / max(1, m.size - int(m.sum()))))
    so[~m, :, 0] = so[~m, :, 0] * corr


def fixed(block, xp=np):
    lut = xp.asarray(FIXED_LUT)
    q = xp.take(lut, xp.asarray(block))            # [T, nchan]
    return xp.ascontiguousarray(q[:, ::-1].T)       # [nchan ascending, T]


def per_channel(block, xp=np, step=2688):
    g = xp.asarray(block)
    T, nchan = g.shape
    q = xp.empty((T, nchan), dtype=xp.uint8)
    for j in range(0, nchan, step):
        x = g[:, j:j + step].astype(xp.float32)
        p16, med, p84 = xp.percentile(x[::4], xp.asarray([16.0, 50.0, 84.0], dtype=xp.float32), axis=0)
        sig = (p84 - p16) / 2
        v = xp.clip(xp.rint((x - med) / xp.maximum(sig / 2, 1e-3)), -7, 7).astype(xp.int8)
        bad = (sig < 1) | (med > 250)
        v = xp.where(bad[None, :] | (g[:, j:j + step] == 255), xp.int8(-8), v)
        q[:, j:j + step] = (v & 0x0F).astype(xp.uint8)
    return xp.ascontiguousarray(q[:, ::-1].T)
