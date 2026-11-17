#!/usr/bin/env python3
"""Loss of RANGE when requantizing: the same B_001 pulse scaled ×1 ... ×100.

Works in the frame aligned at the true DM (each channel shifted by its delay), with the
noise-free model of the generator (same physics) + the same noise realization for all
scales. The quantization is per sample, so quantizing before or after aligning is the same.

  a) transfer function of each quantizer (input value -> stored value), with the
     noise distribution and the per-pixel pulse value for each scale
  b) measured S/N against the true S/N: where each quantization saturates
  c) the aligned pulse at scale ×30 (S/N ~4600): unquantized, 8 bits and 4 bits

Usage (simulator venv): python fig_quantization_range.py   (needs CHARTS_SET_DIR)
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
SCHEMES = [("unquantized", "#8a8984"), ("8 bits (4 lev/σ, up to +55σ)", plt.cm.viridis(0.0)),
           ("4 bits 3 lev/σ (±2.3σ)", plt.cm.viridis(0.3)), ("4 bits 2 lev/σ (±3.5σ)", plt.cm.viridis(0.6)),
           ("4 bits 1 lev/σ (±7σ)", plt.cm.viridis(0.85))]
SCALES = [1, 2, 3, 5, 10, 20, 30, 50, 100]


def q8(x):        # the .fil: floor(32.5 + 4x) clipped to 0-255, returned in sigma
    return (np.clip(np.floor(x * 4 + 32.5), 0, 255) - 32.0) / 4.0


def q4(x, lev):   # from the 8 bits, as the emitter would do
    return np.clip(np.rint(q8(x) * lev), -7, 7) / lev


QUANT = [lambda x: x, q8, lambda x: q4(x, 3), lambda x: q4(x, 2), lambda x: q4(x, 1)]


def aligned_snr(al, nchan):
    """al: aligned (time, channel); S/N with the best boxcar, noise measured far from the pulse."""
    s = al.sum(axis=1)
    noise = np.r_[s[:700], s[-700:]]
    s = s - noise.mean()
    sd = noise.std()
    c = np.concatenate([[0.0], np.cumsum(s)])
    best = 0.0
    for w in G.WIDTHS[G.WIDTHS <= 256]:
        v = (c[w:] - c[:-w]) / (sd * np.sqrt(w))
        best = max(best, float(v[850:1150].max()))
    return best


def main():
    r = next(x for x in csv.DictReader(open(SET / "B_amplitud_fija" / "manifest.csv")) if x["name"] == "B_001")
    p = {k: float(r[k]) for k in ("dm", "width_s", "scat_ref_s", "sp_idx", "sp_run")}
    channels, t_c = GEN.base(p, G.DM_INDEX, float(r["t_top_s"]), int(r["nsamp"]))
    scale = float(r["escala_jy"]) / GEN.SIGMA
    nch = len(channels)
    K = 2000
    # aligned signal: each channel centred on its arrival
    sig = np.zeros((K, nch), np.float32)
    for c, (i0, v) in enumerate(channels):
        ic = int(round(t_c[c] / 1e-3))
        a = i0 - ic + K // 2
        lo, hi = max(0, a), min(K, a + len(v))
        if hi > lo:
            sig[lo:hi, c] = scale * v[lo - a:hi - a]
    noise = np.random.default_rng(7).standard_normal((K, nch), dtype=np.float32)
    peak = float(sig.max())
    print(f"B_001: pulse peak per pixel = {peak:.2f} σ (scale ×1)")

    res = np.zeros((len(SCALES), len(QUANT)))
    for i, e in enumerate(SCALES):
        x = noise + e * sig
        for j, fq in enumerate(QUANT):
            res[i, j] = aligned_snr(fq(x), nch)
        print(f"  ×{e:<4d} peak/pixel {e * peak:6.2f} σ  " + "  ".join(f"{v:9.1f}" for v in res[i]), flush=True)
    snr_true = res[:, 0]

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.38, wspace=0.28, top=0.9)
    fig.suptitle("The range loss of 4 bits: the same pulse (B_001) brighter and brighter",
                 fontsize=15, fontweight="bold", color=TXT, x=0.05, ha="left")

    # a) transfer
    ax = fig.add_subplot(gs[0, 0])
    xx = np.linspace(-6, 12, 4000)
    for (name, col), fq in zip(SCHEMES, QUANT):
        ax.plot(xx, fq(xx), color=col, lw=2.2 if name.startswith("unq") else 1.4, label=name)
    dens = np.exp(-xx ** 2 / 2)
    ax.fill_between(xx, -8, -8 + 2.5 * dens, color="#8a8984", alpha=0.25, lw=0)
    ax.text(0, -7.6, "noise (gaussian)", ha="center", fontsize=8, color=TXT2)
    for e, m in ((1, "×1"), (10, "×10"), (30, "×30")):
        ax.axvline(e * peak, color=TXT2, ls=":", lw=0.8)
        ax.text(e * peak, 10.5, f"peak\n{m}", ha="center", fontsize=8, color=TXT2)
    ax.set_xlim(-6, 12)
    ax.set_ylim(-8, 12)
    ax.set_xlabel("true value of a pixel (σ)")
    ax.set_ylabel("stored value (σ)")
    ax.set_title("a  What each format stores: everything beyond\n    the ceiling is clipped", loc="left",
                 fontsize=11, fontweight="bold", color=TXT)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(color=GRID, lw=0.6)

    # b) measured vs true S/N
    ax = fig.add_subplot(gs[0, 1:])
    for j, (name, col) in enumerate(SCHEMES):
        ax.plot(snr_true, res[:, j] / snr_true * 100, "o-", color=col, lw=1.8, ms=6, label=name)
    for i, e in enumerate(SCALES):
        ax.text(snr_true[i], 101.2, f"×{e}\n{e * peak:.1f}σ", ha="center", fontsize=7.5, color=TXT2)
    ax.set_xscale("log")
    ax.set_ylim(55, 104)
    ax.axhline(100, color=TXT2, lw=0.8)
    ax.set_xlabel("true S/N of the pulse (unquantized)")
    ax.set_ylabel("measured S/N / true S/N (%)")
    ax.set_title("b  S/N kept versus brightness (top: scale and peak per pixel)", loc="left",
                 fontsize=11, fontweight="bold", color=TXT)
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    ax.grid(color=GRID, lw=0.6)

    # c) aligned pulse ×30
    e = 30
    x = noise + e * sig
    i_snr = SCALES.index(e)
    for j, ((name, col), fq) in enumerate(zip(SCHEMES[:3], QUANT[:3])):
        ax = fig.add_subplot(gs[1, j])
        al = fq(x)[K // 2 - 120:K // 2 + 120]
        sub = al.reshape(al.shape[0], 168, 128).mean(axis=2)
        ax.imshow(sub.T, aspect="auto", cmap="viridis", vmin=-0.3, vmax=1.6 * e * peak / 4,
                  extent=[-120, 120, 300, 501.7], origin="lower")
        ax.set_title(f"c{j + 1}  ×{e}: {name.split(' (')[0]}\n     measured S/N {res[i_snr, j]:.0f}",
                     loc="left", fontsize=11, fontweight="bold", color=TXT)
        ax.set_xlabel("ms (aligned at the DM)")
        if j == 0:
            ax.set_ylabel("MHz")
    fig.text(0.05, 0.015,
             f"Pulse peak per pixel at scale ×1: {peak:.2f}σ. With 4 bits and 3 levels/σ the pulse is clipped when "
             "its per-pixel value exceeds ~2σ: in c3 the bright part (above 450 MHz) is flattened at the ceiling.",
             fontsize=10, color=TXT2)
    out = HERE / "plots_21504" / "quantization_range_B_001.png"
    fig.savefig(out, dpi=105, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
