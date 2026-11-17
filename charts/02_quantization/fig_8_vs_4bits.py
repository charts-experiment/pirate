#!/usr/bin/env python3
"""Comparison figure: the set's 8 bits against 4 bits with n = 1, 2 and 3 levels per sigma.

Left: the staircase of each conversion (reconstructed value, in sigma, against the input signal),
with the input noise (bottom) and the output noise (left). Right: S/N measured by pirate
against the true S/N of a 10 ms full-band burst (all its pixels with the same height).
Same theory as quantization_theory.py: 8-bit level L = floor(32.5 + 4x),
clipped to 0-255; 4 bits = clip(floor((L - 32) * n/4), -7, 7), reconstructed at the centre of each code
(like the emitter: with n = 2 each code groups 2 counts, centre 32.5).

Usage: python fig_8_vs_4bits.py  ->  plots_21504/bits_comparison.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

OUT = Path(__file__).resolve().parent / "plots_21504"
OUT.mkdir(exist_ok=True)
L = np.arange(256)
EDGES = (np.arange(257) - 32.5) / 4.0             # input (in sigma) where each level L starts
EDGES[0], EDGES[-1] = -np.inf, np.inf
N_PIX = 21504 * 10                                 # reference burst: the whole band for 10 ms
THRESHOLD = 6.5                                    # pirate operating threshold (10 noise FP per beam·hour)


def four_bits(n):
    """Reconstructed value (in sigma) for each level L: code q = clip(floor((L - 32)·n/4), -7, 7)."""
    q = np.clip(np.floor((L - 32) * n / 4.0), -7, 7)
    return (q + 0.5) / n


OPTIONS = [                                        # (label, reconstructed value per level L, ceiling in sigma)
    ("8 bits (the .fil)", (L - 32) / 4.0, "+55σ"),
    ("4 bits, n = 3", four_bits(3), "±2.3σ"),
    ("4 bits, n = 2 (in use)", four_bits(2), "±3.5σ"),
    ("4 bits, n = 1", four_bits(1), "±7σ"),
]
COLOURS = plt.cm.viridis(np.linspace(0.0, 0.85, len(OPTIONS)))


def level_probs(s):
    """P(L) for N(0,1) noise plus a signal s (in sigma)."""
    return np.diff(norm.cdf(EDGES - s))


def snr_fraction(g, s):
    """Measured S/N / true S/N for a pixel with signal s: (E[g|s] - E[g|0]) / (s * σ_g)."""
    p0 = level_probs(0.0)
    m0 = (p0 * g).sum()
    sd = np.sqrt((p0 * (g - m0) ** 2).sum())
    return np.array([((level_probs(si) * g).sum() - m0) / (si * sd) for si in s])


def main():
    plt.rcParams.update({"font.size": 15})
    fig = plt.figure(figsize=(17, 6.6), facecolor="white")
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.12, 1.9], wspace=0.25, hspace=0.42)

    x = np.linspace(-4.5, 8.5, 4001)
    level = np.clip(np.floor(32.5 + 4 * x), 0, 255).astype(int)
    noise = norm.pdf(x)
    p0 = level_probs(0.0)
    for i, ((name, g, ceiling), c) in enumerate(zip(OPTIONS, COLOURS)):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        # Input noise (the same in the four panels), on the horizontal axis.
        ax.fill_between(x, -8, -8 + noise * 10, color="0.85", lw=0)
        # The same noise after the conversion (what pirate sees), on the vertical axis:
        # one bar per possible value, with length proportional to the probability density.
        vals, inv = np.unique(g, return_inverse=True)
        prob = np.bincount(inv, weights=p0)
        step = np.min(np.diff(vals))
        ax.barh(vals, prob / step * 6, height=step * 0.9, left=-4.5, color=c, alpha=0.45, lw=0)
        clipped = prob[0] + prob[-1]
        ax.plot(x, x, color="0.6", lw=1, ls=":")
        ax.plot(x, g[level], color=c, lw=2.2)
        ax.set_xlim(-4.5, 8.5)
        ax.set_ylim(-8, 8.5)
        txt_clip = f"{100 * clipped:.2f} % of the noise at the ceiling" if clipped >= 1e-4 else "nothing at the ceiling"
        ax.set_title(f"{name}\nceiling {ceiling} · {txt_clip}", fontsize=13)
        if i == 0:
            ax.text(1.2, -7.6, "input noise\n(the same in all)", fontsize=11, color="0.4")
            ax.text(-4.3, 5.2, "output noise", fontsize=11, color=c, rotation=0)
        if i // 2 == 1:
            ax.set_xlabel("input signal (σ)")
        if i % 2 == 0:
            ax.set_ylabel("value pirate sees (σ)")

    # Right: measured S/N against the true S/N of the full burst (power integrated over its pixels).
    # Reference burst: the whole band (21 504 channels) for 10 ms, all pixels with the same
    # height s; true S/N = s * sqrt(N_PIX).
    ax = fig.add_subplot(gs[:, 3])
    sqrt_npix = np.sqrt(N_PIX)
    snr = np.logspace(0, 5, 400)
    s = snr / sqrt_npix
    ax.axvspan(5, 100, color="0.92", lw=0)
    ax.text(22, 250, "bursts\nof the set", ha="center", fontsize=13, color="0.35")
    ax.axhline(THRESHOLD, color="0.45", lw=1.2, ls="--")
    ax.text(1.15, THRESHOLD * 1.18, f"detection threshold ({THRESHOLD:.1f})", fontsize=12, color="0.35")
    ax.plot(snr, snr, color="0.6", lw=1, ls=":")
    for (name, g, _), c in zip(OPTIONS, COLOURS):
        f = snr_fraction(g, s)
        ceiling = f"{snr[-1] * f[-1]:,.0f}".replace(",", " ")
        ax.plot(snr, snr * f, color=c, lw=2.6, label=f"{name}: ceiling {ceiling}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e5)
    ax.set_ylim(1, 1e5)
    ax.set_xlabel("true S/N of the burst (integrated power)")
    ax.set_ylabel("S/N measured by pirate")
    ax.set_title("10 ms full-band burst: measured versus true S/N", fontsize=15)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=12.5, frameon=False, title="maximum measurable S/N", title_fontsize=12.5)
    top = ax.secondary_xaxis("top", functions=(lambda v: v / sqrt_npix, lambda v: v * sqrt_npix))
    top.set_xlabel("signal in each pixel (σ)", fontsize=12)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "bits_comparison.png", dpi=110, bbox_inches="tight")
    for name, g, ceiling in OPTIONS:
        f = snr_fraction(g, np.array([0.01, 1.0, 4.0, 8.0, 24.0]))
        print(f"{name:24s} ceiling {ceiling:6s} S/N: " + "  ".join(f"{v:.3f}" for v in f) + "   (s = 0.01 / 1 / 4 / 8 / 24 σ)")


if __name__ == "__main__":
    main()
