#!/usr/bin/env python3
"""Quantization 8 bits -> 4 bits expressed in COUNTS (the integers that are stored).

  a) histogram of the noise of one channel in 8-bit counts (floor 32, 1 sigma = 4 counts)
  b) conversion from 8-bit count to 4-bit code (-7..+7) for n = 1, 2 and 3 levels per sigma
  c) S/N kept versus the pulse height per pixel, in 8-bit counts above the floor
  d) the pixels of a bright pulse (B_001 ×10) in 8-bit counts and as 4-bit codes (n = 2)

Usage (simulator venv): python fig_quantization_counts.py   (needs CHARTS_SET_DIR)
"""
import os
import sys
import csv
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
(HERE / "plots_21504").mkdir(exist_ok=True)
SIM = HERE.parent / "01_simulator"                   # simulator generator and physics
spec = importlib.util.spec_from_file_location("gen", SIM / "12_generate_comparison_set_21504.py")
GEN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GEN)
G = GEN.G
SET = os.environ.get("CHARTS_SET_DIR_TEST") or os.environ.get("CHARTS_SET_DIR")
if not SET:
    sys.exit("ERROR: export CHARTS_SET_DIR=/folder/of/group3 (or CHARTS_SET_DIR_TEST for a small set)")
SET = Path(SET)
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {8: plt.cm.viridis(0.0), 3: plt.cm.viridis(0.3), 2: plt.cm.viridis(0.6), 1: plt.cm.viridis(0.85)}


def code4(c8, n):
    """4-bit code from the 8-bit count. n = 2 with centre 32.5 (no ties)."""
    centre = 32.5 if n == 2 else 32.0
    return np.clip(np.rint((c8 - centre) * n / 4.0), -7, 7)


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GRID, "axes.labelcolor": TXT2, "xtick.color": TXT2, "ytick.color": TXT2})
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.3, top=0.9)
    fig.suptitle("Quantization in counts: 8 bits (0–255) of the .fil  →  4-bit code (−7…+7) for pirate",
                 fontsize=15, fontweight="bold", color=TXT, x=0.05, ha="left")

    # a) noise in counts
    ax = fig.add_subplot(gs[0, 0])
    c = np.arange(0, 70)
    p = norm.cdf(c + 1 - 32.5, scale=4) - norm.cdf(c - 32.5, scale=4)
    ax.bar(c, 100 * p, width=0.85, color=COL[8])
    for k, s in ((-2, "−2σ"), (-1, "−1σ"), (1, "+1σ"), (2, "+2σ"), (3, "+3σ")):
        ax.axvline(32 + 4 * k, color=TXT2, ls=":", lw=0.8)
        ax.text(32 + 4 * k, 10.6, s, ha="center", fontsize=8, color=TXT2)
    ax.axvline(32, color=TXT, lw=1)
    ax.text(32, 11.4, "floor = 32", ha="center", fontsize=9, color=TXT, fontweight="bold")
    ax.set_xlim(10, 56)
    ax.set_ylim(0, 12)
    ax.set_xlabel("8-bit count")
    ax.set_ylabel("% of the noise samples")
    ax.set_title("a  Noise of one channel in the .fil\n    1σ = 4 counts", loc="left", fontsize=11, fontweight="bold",
                 color=TXT)

    # b) conversion 8 -> 4 bits
    ax = fig.add_subplot(gs[0, 1:])
    c8 = np.arange(0, 90)
    ax.axvspan(28, 36, color="#8a8984", alpha=0.12, lw=0)
    ax.axvspan(24, 40, color="#8a8984", alpha=0.08, lw=0)
    ax.text(32, -8.6, "68 % of the noise\nbetween 28 and 36", ha="center", fontsize=8, color=TXT2)
    for n, lab in ((1, "n = 1: 1 level/σ (4 counts per code)"), (2, "n = 2: 2 levels/σ (2 counts per code)"),
                   (3, "n = 3: 3 levels/σ (~1.33 counts per code)")):
        q = code4(c8, n)
        ax.step(c8, q, where="mid", color=COL[n], lw=2, label=lab)
        ceiling = int(c8[np.argmax(q >= 7)])
        ax.plot(ceiling, 7, "o", color=COL[n], ms=6)
        ax.annotate(f"from {ceiling} counts\n7 is stored", xy=(ceiling, 7), xytext=(ceiling + 3, 4.6 - (n - 2) * 1.6),
                    fontsize=8, color=COL[n], arrowprops=dict(arrowstyle="->", color=COL[n], lw=0.7))
    ax.axhline(7, color=TXT2, lw=0.6, ls="--")
    ax.axhline(-7, color=TXT2, lw=0.6, ls="--")
    ax.set_xlim(0, 90)
    ax.set_ylim(-9.5, 9)
    ax.set_yticks(range(-8, 9, 2))
    ax.set_xlabel("8-bit count (what is in the .fil)")
    ax.set_ylabel("4-bit code (what pirate receives)")
    ax.set_title("b  Count-by-count conversion: 15 possible codes (−8 is reserved for masked samples)",
                 loc="left", fontsize=11, fontweight="bold", color=TXT)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(color=GRID, lw=0.6)
    sec = ax.secondary_xaxis("top", functions=(lambda x: (x - 32) / 4, lambda s: 32 + 4 * s))
    sec.set_xlabel("in σ (reference)", color=TXT2)

    # c) S/N kept vs pulse height in counts (measured with fig_quantization_range.py, B_001)
    ax = fig.add_subplot(gs[1, 0])
    peak_sigma = np.array([0.80, 1.60, 2.41, 4.01, 8.02, 16.05, 24.07, 40.12, 80.24])
    snr = {"true": [153.3, 306.7, 460.1, 766.8, 1534.0, 3068.4, 4602.7, 7671.5, 15343.4],
           8: [153.2, 306.5, 459.7, 766.0, 1532.4, 3065.2, 4598.1, 7663.8, 14104.4],
           3: [150.9, 294.3, 421.9, 606.4, 816.5, 988.0, 1078.6, 1184.7, 1318.1],
           2: [152.0, 303.4, 451.7, 714.8, 1066.6, 1340.1, 1480.1, 1644.3, 1848.4],
           1: [146.9, 294.7, 442.2, 737.6, 1436.5, 2110.9, 2426.7, 2783.8, 3217.7]}
    true = np.array(snr["true"])
    counts = 4 * peak_sigma
    for k, lab in ((8, "8 bits"), (2, "4 bits n = 2"), (3, "4 bits n = 3"), (1, "4 bits n = 1")):
        ax.plot(counts, 100 * np.array(snr[k]) / true, "o-", color=COL[k], lw=1.8, ms=5, label=lab)
    for x, s in zip(counts[[0, 3, 4, 6]], true[[0, 3, 4, 6]]):
        ax.text(x, 103, f"S/N\n{s:.0f}", ha="center", fontsize=7.5, color=TXT2)
    ax.set_xscale("log")
    ax.set_ylim(0, 110)
    ax.axhline(100, color=TXT2, lw=0.8)
    ax.set_xlabel("pulse height per pixel\n(8-bit counts above the floor)")
    ax.set_ylabel("S/N kept (%)")
    ax.set_title("c  B_001 brighter and brighter", loc="left", fontsize=11, fontweight="bold", color=TXT)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.grid(color=GRID, lw=0.6)

    # d) pixels of the ×10 pulse in counts
    r = next(x for x in csv.DictReader(open(SET / "B_amplitud_fija" / "manifest.csv")) if x["name"] == "B_001")
    p = {k: float(r[k]) for k in ("dm", "width_s", "scat_ref_s", "sp_idx", "sp_run")}
    channels, t_c = GEN.base(p, G.DM_INDEX, float(r["t_top_s"]), int(r["nsamp"]))
    scale = float(r["escala_jy"]) / GEN.SIGMA * 10
    ch = np.arange(len(channels) - 40, len(channels))                    # 40 fine channels near 501 MHz
    K = 60
    sig = np.zeros((K, ch.size), np.float32)
    for j, cidx in enumerate(ch):
        i0, v = channels[cidx]
        ic = int(round(t_c[cidx] / 1e-3))
        a = i0 - ic + K // 2
        lo, hi = max(0, a), min(K, a + len(v))
        if hi > lo:
            sig[lo:hi, j] = scale * v[lo - a:hi - a]
    x = np.random.default_rng(3).standard_normal(sig.shape).astype(np.float32) + sig
    c8 = np.clip(np.floor(x * 4 + 32.5), 0, 255)
    q2 = code4(c8, 2)
    for k, (dat, tit, vmin, vmax, cl) in enumerate(((c8, "8-bit count", 16, 80, "counts (8 bits)"),
                                                    (q2, "4-bit code (n = 2)", -7, 7, "code (4 bits)"))):
        ax = fig.add_subplot(gs[1, 1 + k])
        im = ax.imshow(dat.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest",
                       extent=[-K // 2, K // 2, 0, ch.size])
        ax.set_xlabel("ms (aligned at the DM)")
        ax.set_ylabel("fine channel (near 501 MHz)")
        n_ceiling = int((dat >= (7 if k else 255)).sum())
        ax.set_title(f"d{k + 1}  B_001 ×10 (S/N ~1500): {tit}\n     maximum {dat.max():.0f}" +
                     (f"  ·  {n_ceiling} pixels clipped at 7" if k else ""),
                     loc="left", fontsize=11, fontweight="bold", color=TXT)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label(cl)
    fig.text(0.05, 0.02, "With n = 2 each 4-bit code groups 2 counts of 8 bits: 32–33 → 0, 34–35 → +1, … and from "
             "46 counts (+3.5σ) everything stays at 7. In d2 the pulse loses its shape: the core (≈46–65 counts in d1) "
             "is flat at 7.", fontsize=10, color=TXT2)
    out = HERE / "plots_21504" / "quantization_in_counts.png"
    fig.savefig(out, dpi=105, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
