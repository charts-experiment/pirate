#!/usr/bin/env python3
"""Theoretical S/N efficiency of each quantization (weak signal in gaussian noise).

Q(x) = reconstructed value of the sample x ~ N(0,1) (in sigma). For a weak signal s:
  E[Q(x+s)] - E[Q(x)] ~ s * sum_k J_k phi(t_k)      (t_k thresholds, J_k jump of Q at t_k)
  eta = sum_k J_k phi(t_k) / sqrt(Var Q(x))          fraction of S/N that is kept.
Computed exactly from the thresholds of the set's 8-bit quantizer
(level L = floor(32.5 + 4x), clipped to 0-255) and from each 4-bit mapping from L.
"""
import numpy as np
from scipy.stats import norm

L = np.arange(256)
# 8 bits: L = floor(32.5 + 4x)  ->  threshold between L-1 and L at x = (L - 32.5)/4
t8 = (L[1:] - 32.5) / 4.0
p8 = np.diff(np.r_[0.0, norm.cdf(t8), 1.0])              # P(L)


def eta_from_8bits(g):
    """g: reconstructed value (in sigma) for each 8-bit level L."""
    g = np.asarray(g, float)
    m = (p8 * g).sum()
    var = (p8 * (g - m) ** 2).sum()
    gain = (np.diff(g) * norm.pdf(t8)).sum()
    return gain / np.sqrt(var), var


def eta_direct(step, nmax):
    """Centred uniform quantizer applied to the float: q = clip(rint(x/step), -nmax, nmax)."""
    k = np.arange(-nmax, nmax + 1)
    t = (k[:-1] + 0.5) * step
    p = np.diff(np.r_[0.0, norm.cdf(t), 1.0])
    v = k * step
    var = (p * v ** 2).sum() - (p * v).sum() ** 2
    return (step * norm.pdf(t)).sum() / np.sqrt(var)


rows = [("8 bits (set: 4 levels/σ, floor 32)", eta_from_8bits((L - 32) / 4.0)[0])]
for n, name in ((2, "4b 2lev fixed"), (1, "4b 1lev fixed"), (3, "4b 3lev fixed")):
    rows.append((f"{name} (from 8 bits)", eta_from_8bits(np.clip(np.rint((L - 32) / 4.0 * n), -7, 7) / n)[0]))
rows.append(("4b 2lev from float (without going through 8 bits)", eta_direct(0.5, 7)))
print("Theoretical S/N efficiency (weak signal), relative to the unquantized data:")
for name, e in rows:
    print(f"  {name:50s} {e:.4f}   loss {100 * (1 - e):.2f} %")

print("\nReference (optimal uniform quantizer, weak signal):")
for bits in (1, 2, 3, 4):
    nmax = 2 ** (bits - 1) - 1 if bits > 1 else None
    if bits == 1:
        print(f"  1 bit: {np.sqrt(2 / np.pi):.4f}")
        continue
    steps = np.linspace(0.05, 2.0, 2000)
    # with 2^bits - 1 symmetric levels (like int4 with -8 reserved)
    e = [eta_direct(p, nmax) for p in steps]
    i = int(np.argmax(e))
    print(f"  {bits} bits ({2 * nmax + 1} levels): max η {e[i]:.4f} with step {steps[i]:.3f} σ "
          f"({1 / steps[i]:.2f} levels/σ)")

print("\npirate (scale/offset per channel and block of 256 samples):")
print("  - the offset is estimated with the block mean: a pulse of width w (samples) inside a")
print("    block loses ~w/256 of its amplitude; at 21 504 channels the pulse per channel lasts ~intrinsic width + scattering.")
for w in (1, 4, 16, 32, 64, 128):
    print(f"    w = {w:4d} ms -> S/N x {1 - w / 256:.3f}")
print("  - the error of the mean (sigma/sqrt(256)) is a 256 ms step per channel: it adds ~1/256 of variance (negligible).")
