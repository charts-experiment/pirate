#!/usr/bin/env python3
"""Explanatory figure of the quantization (the set's 8 bits -> 4 bits for pirate).

1) What each quantization does to the noise of one channel (histograms of values).
2) The dedispersed pulse with each quantization.
3) How much S/N is kept: theory (bar) and measured in the bright bursts of B (points).

Uses B_001 of the test set (rebuilt in float with the same seed) and
plots_21504/quantization_21504.csv (from quantization_21504.py).
Usage (simulator venv): python fig_quantization_intuitive.py   (needs CHARTS_SET_DIR)
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
spec = importlib.util.spec_from_file_location("cq", HERE / "quantization_21504.py")
CQ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CQ)

SET = os.environ.get("CHARTS_SET_DIR_TEST") or os.environ.get("CHARTS_SET_DIR")
if not SET:
    sys.exit("ERROR: export CHARTS_SET_DIR=/folder/of/group3 (or CHARTS_SET_DIR_TEST for a small set)")
SET = Path(SET)
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GREY = "#8a8984"
SCHEMES = [  # (key in the csv, label, colour)
    ("8 bits", "8 bits (the .fil)\n4 levels/σ", plt.cm.viridis(0.0)),
    ("4b 3lev fixed", "4 bits, 3 levels/σ\n(recommended)", plt.cm.viridis(0.25)),
    ("4b 2lev fixed", "4 bits, 2 levels/σ", plt.cm.viridis(0.5)),
    ("4b 1lev fixed", "4 bits, 1 level/σ", plt.cm.viridis(0.72)),
    ("4b pirate", "pirate-style 4 bits\n(offset every 256 ms)", plt.cm.viridis(0.9)),
]
THEORY = {"8 bits": 99.74, "4b 3lev fixed": 99.03, "4b 2lev fixed": 98.22, "4b 1lev fixed": 95.35, "4b pirate": 98.22}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TXT2, "xtick.color": TXT2,
                     "ytick.color": TXT2, "axes.titlecolor": TXT, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
                     "axes.facecolor": "white"})


def quantize(d8, lev):
    return np.clip(np.rint(d8 / 4.0 * lev), -7, 7) / lev


def main():
    r = next(x for x in csv.DictReader(open(SET / "B_amplitud_fija" / "manifest.csv")) if x["name"] == "B_001")
    xf = CQ.rebuild_float(r)[:, ::-1]                              # (time, descending channel), in sigma
    u8 = np.asarray(CQ.VAL.read_fil(r["fil_path"]))
    d8 = u8.astype(np.float32) - 32.0                              # levels (1 sigma = 4)

    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.62, wspace=0.32, top=0.86)
    fig.suptitle("What is lost when going from 8 bits to 4 bits?  (group-3 test set, 21 504 channels)",
                 fontsize=14, fontweight="bold", color=TXT, x=0.06, ha="left", y=0.995)
    fig.text(0.06, 0.945, "1  The noise of each channel, quantized (histogram of values; line = ideal gaussian)",
             fontsize=12, fontweight="bold", color=TXT)
    fig.text(0.06, 0.475, "2  What the search sees: the dedispersed pulse", fontsize=12, fontweight="bold", color=TXT)
    fig.text(0.53, 0.475, "3  How much S/N is kept", fontsize=12, fontweight="bold", color=TXT)

    # ---- 1) noise histograms ----
    noise_f = xf[:2000, ::64].ravel()
    noise_8 = d8[:2000, ::64].ravel() / 4.0
    panels = [("float", "unquantized", GREY, noise_f, None),
              ("8 bits", "8 bits: 4 levels/σ", SCHEMES[0][2], noise_8, 0.25),
              ("4b 3lev fixed", "4 bits: 3 levels/σ", SCHEMES[1][2], quantize(d8[:2000, ::64].ravel(), 3), 1 / 3),
              ("4b 1lev fixed", "4 bits: 1 level/σ", SCHEMES[3][2], quantize(d8[:2000, ::64].ravel(), 1), 1.0)]
    xx = np.linspace(-4.5, 4.5, 400)
    for k, (_, tit, col, v, step) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        if step is None:
            ax.hist(v, bins=120, range=(-4.5, 4.5), density=True, color=col, alpha=0.8)
        else:
            vals, cnt = np.unique(np.round(v / step).astype(int), return_counts=True)
            ax.bar(vals * step, cnt / cnt.sum() / step, width=step * 0.8, color=col)
            if k >= 2:
                ceiling = 7 * step
                ax.axvspan(ceiling, 4.5, color=col, alpha=0.08)
                ax.axvspan(-4.5, -ceiling, color=col, alpha=0.08)
                ax.annotate(f"everything beyond ±{ceiling:.1f}σ\ngoes to the last level",
                            xy=(ceiling, 0.05), xytext=(0.6, 0.93), textcoords="axes fraction", fontsize=8,
                            color=TXT2, arrowprops=dict(arrowstyle="->", color=TXT2, lw=0.6))
        ax.plot(xx, norm.pdf(xx), color=TXT, lw=1, ls="--")
        n_val = "continuous values" if step is None else f"{len(np.unique(np.round(v / step)))} distinct values"
        ax.set_title(f"{tit}\n{n_val}", fontsize=10)
        ax.set_xlim(-4.5, 4.5)
        ax.set_xlabel("noise of one channel (σ)")
        ax.set_yticks([])
        if k == 2:
            ax.text(0.5, -0.36, "the \"teeth\": the 8-bit grid (σ/4) does not match the 4-bit one (σ/3);\n"
                    "each level receives 1 or 2 8-bit levels. Already included in the theory.",
                    transform=ax.transAxes, ha="center", fontsize=8, color=TXT2)

    # ---- 2) dedispersed series of the whole band ----
    ax = fig.add_subplot(gs[1, :2])
    f = CQ.VAL.freqs_fil({"freq_min_mhz": 299.8545, "freq_max_mhz": 501.7455})
    dm = float(r["dm"])
    versions = [("unquantized", xf, GREY, 2.6), ("8 bits", d8 / 4.0, SCHEMES[0][2], 1.3),
                ("4 bits, 3 levels/σ", quantize(d8, 3), SCHEMES[1][2], 1.3),
                ("4 bits, 1 level/σ", quantize(d8, 1), SCHEMES[3][2], 1.3)]
    for name, x, col, lw in versions:
        series, off = CQ.VAL.dedisperse(x, f, dm, CQ.G.DM_INDEX)
        i = int(round(float(r["t_top_s"]) / 1e-3)) + off
        noise = series[:i - 2000]
        z = (series - np.median(noise)) / noise.std()
        k = np.arange(-80, 80)
        snr = CQ.snr(series, i, x.shape[1])
        ax.plot(k, z[i + k], color=col, lw=lw, label=f"{name}: S/N {snr:.1f}")
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xlabel("ms relative to the expected arrival at 501.6 MHz")
    ax.set_ylabel("S/N per 1 ms sample")
    ax.set_title("B_001 (DM 727, 21 504 channels summed at the true DM)", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.text(0.02, 0.92, "the curves almost coincide:\nthe difference is ~1–5 % of the peak", transform=ax.transAxes,
            fontsize=9, color=TXT2, va="top")

    # ---- 3) S/N kept ----
    ax = fig.add_subplot(gs[1, 2:])
    ax.tick_params(axis="y", pad=4)
    rows = list(csv.DictReader(open(HERE / "plots_21504" / "quantization_21504.csv")))
    bright = [x for x in rows if float(x["float"]) > 50]
    y = np.arange(len(SCHEMES))[::-1]
    for yk, (key, lab, col) in zip(y, SCHEMES):
        ax.barh(yk, THEORY[key] - 90, left=90, height=0.55, color=col, alpha=0.35)
        med = np.array([100 * float(x[key]) / float(x["float"]) for x in bright])
        ax.plot(med, np.full(med.size, yk), "o", color=col, ms=8, mec="white", mew=1.2, zorder=3)
        ax.text(THEORY[key] + 0.12, yk + 0.31, f"{THEORY[key]:.1f} %", va="center", fontsize=9, color=TXT)
    ax.set_yticks(y, [e for _, e, _ in SCHEMES])
    ax.set_xlim(90, 101)
    ax.set_xlabel("S/N kept relative to the unquantized data (%)")
    ax.axvline(100, color=TXT2, lw=0.8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("bar = theory  ·  points = measured in the 4 bright bursts of B (S/N 57–156)", fontsize=10)
    ax.annotate("it also loses ~width/256 in wide pulses:\nA_007 (scattering 26 ms) lost 22 %",
                xy=(97.0, y[-1]), xytext=(90.3, y[-1] - 0.75), fontsize=8, color=TXT2,
                arrowprops=dict(arrowstyle="->", color=TXT2, lw=0.6))
    ax.set_ylim(-1.1, len(SCHEMES) - 0.4)

    out = HERE / "plots_21504" / "quantization_explained.png"
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
