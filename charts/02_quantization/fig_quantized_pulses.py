#!/usr/bin/env python3
"""What a pulse looks like with each quantization (same noise and same signal).

Rows: unquantized · 8 bits (the .fil) · 4 bits 3 levels/σ · 4 bits 1 level/σ.
Columns:
  1) sweep: upper part of the band (where B_001 emits), averaged to 64 channels × 4 ms
  2) pulse aligned at the true DM: each channel shifted by its delay, 168 subbands × 1 ms
  3) raw pixels: 48 fine channels (9.39 kHz) × 120 ms at full resolution, not averaged

Usage (simulator venv): python fig_quantized_pulses.py [NAME=B_001]   (needs CHARTS_SET_DIR)
"""
import os
import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
(HERE / "plots_21504").mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("cq", HERE / "quantization_21504.py")
CQ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CQ)
G, VAL = CQ.G, CQ.VAL
SET = os.environ.get("CHARTS_SET_DIR_TEST") or os.environ.get("CHARTS_SET_DIR")
if not SET:
    sys.exit("ERROR: export CHARTS_SET_DIR=/folder/of/group3 (or CHARTS_SET_DIR_TEST for a small set)")
SET = Path(SET)
TXT, TXT2 = "#0b0b0b", "#52514e"


def quantize(d8, lev):
    return np.clip(np.rint(d8 / 4.0 * lev), -7, 7) / lev


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "B_001"
    folder = "B_amplitud_fija" if name.startswith("B") else "A_snr_fijo"
    r = next(x for x in csv.DictReader(open(SET / folder / "manifest.csv")) if x["name"] == name)
    xf = CQ.rebuild_float(r)[:, ::-1]
    d8 = np.asarray(VAL.read_fil(r["fil_path"])).astype(np.float32) - 32.0
    f = VAL.freqs_fil({"freq_min_mhz": 299.8545, "freq_max_mhz": 501.7455})        # descending
    dm, t_top = float(r["dm"]), float(r["t_top_s"])
    delay = G.delay_s(dm, G.DM_INDEX, f) - G.delay_s(dm, G.DM_INDEX, 501.6)
    sh = np.round(delay / 1e-3).astype(int)
    i_top = int(round(t_top / 1e-3))

    versions = [("unquantized", xf), ("8 bits (the .fil)\n4 levels/σ", d8 / 4.0),
                ("4 bits\n3 levels/σ", quantize(d8, 3)), ("4 bits\n1 level/σ", quantize(d8, 1))]

    # windows
    c_hi = slice(0, 8192)                                            # 501.7 -> ~424 MHz (upper part)
    t0b, t1b = i_top - 300, i_top + sh[8191] + 300                   # sweep of the upper part
    k = np.arange(-150, 150)                                         # around the aligned pulse
    c480 = int(np.argmin(np.abs(f - 480.0)))
    cz = np.arange(c480 - 24, c480 + 24)
    tz = i_top + sh[c480] + np.arange(-60, 60)

    fig, axs = plt.subplots(4, 3, figsize=(16, 15), gridspec_kw={"width_ratios": [1.5, 1, 1], "hspace": 0.28,
                                                                  "wspace": 0.18})
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{name}: the same pulse (DM {dm:.0f}, width {float(r['width_s']) * 1e3:.1f} ms, "
                 f"S/N {float(r['snr_esperado']):.0f}) with each quantization", fontsize=15, fontweight="bold",
                 color=TXT, x=0.06, ha="left", y=0.995)
    for row, (lab, x) in enumerate(versions):
        # 1) averaged sweep
        ax = axs[row, 0]
        w = x[t0b:t1b, c_hi]
        nt = (w.shape[0] // 4) * 4
        wf = w[:nt].reshape(nt // 4, 4, 128, 64).mean(axis=(1, 3))
        sig = 1 / np.sqrt(4 * 64)
        ax.imshow(wf.T / sig, aspect="auto", cmap="viridis", vmin=-2, vmax=6,
                  extent=[t0b * 1e-3, (t0b + nt) * 1e-3, f[8191], f[0]])
        ax.set_ylabel(lab, fontsize=12, fontweight="bold", color=TXT, rotation=0, ha="right", va="center",
                      labelpad=12)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_yticks([430, 450, 470, 490])
        # 2) aligned at the DM
        ax = axs[row, 1]
        idx = i_top + sh[None, :] + k[:, None]                       # (t, channel)
        al = x[idx, np.arange(x.shape[1])[None, :]]
        al = al.reshape(len(k), 168, 128).mean(axis=2)
        ax.imshow(al.T / (1 / np.sqrt(128)), aspect="auto", cmap="viridis", vmin=-2, vmax=6,
                  extent=[k[0], k[-1], f[-1], f[0]])
        ax.set_yticks([300, 350, 400, 450, 500])
        # 3) raw pixels
        ax = axs[row, 2]
        z = x[np.ix_(tz, cz)]
        im = ax.imshow(z.T, aspect="auto", cmap="viridis", vmin=-2.5, vmax=2.5, interpolation="nearest",
                       extent=[-60, 60, f[cz[-1]], f[cz[0]]])
        vals = np.unique(np.round(z * 12) / 12).size
        ax.text(0.02, 0.04, f"{'continuous' if row == 0 else f'{vals} distinct values'}", transform=ax.transAxes,
                fontsize=9, color="white", fontweight="bold")
    titles = ["1  Sweep (424–501 MHz; 64 channels × 4 ms)",
              "2  Aligned at the true DM",
              "3  Raw pixels (48 fine channels)"]
    for j, t in enumerate(titles):
        axs[0, j].set_title(t, fontsize=11, fontweight="bold", color=TXT, loc="left")
    axs[3, 0].set_xlabel("time from the start of the file (s)")
    axs[3, 1].set_xlabel("ms relative to the arrival (aligned)")
    axs[3, 2].set_xlabel("ms relative to the arrival at 480 MHz")
    for a in axs[:, 1:].ravel():
        a.tick_params(labelsize=8)
    axs[0, 1].set_ylabel("MHz", fontsize=9)
    axs[0, 2].set_ylabel("MHz", fontsize=9)
    cb = fig.colorbar(im, ax=axs[:, 2], fraction=0.04, pad=0.02)
    cb.set_label("column 3: value of each pixel (σ of one channel)")
    fig.text(0.06, 0.005, "Columns 1 and 2 in σ of the average (scale -2 to 6). In column 3 the pulse is not visible in a "
             "single pixel: per channel and sample it contributes ~0.4σ; it only appears when channels are summed (columns 1 and 2).",
             fontsize=10, color=TXT2)
    out = HERE / "plots_21504" / f"quantized_pulses_{name}.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
