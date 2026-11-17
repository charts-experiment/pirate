#!/usr/bin/env python3
"""Site baseband before and after the ×32 channelization, one figure per antenna (inputs 0–7), both at 1 ms.

Before: 672 coarse channels of 0.3 MHz. Their power at 1 ms is obtained by summing the 32 fine channels of each
(Parseval: with a rectangular FFT, Σ_k |X_k|² = 32·Σ_n |x_n|², exactly).
After: 21 504 channels of 9.375 kHz (01_dynamic_spectrum.py).
Output: figures_upchannelization/antenna{0..7}.png
Usage: python 05_fig_upchannelization.py [INPUT ...]   (no arguments: all 8)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
D = np.load(HERE / "dynamic_spectrum.npy", mmap_mode="r")
f = np.load(HERE / "frequencies_mhz.npy")
fg = 300.0 + 0.3 * np.arange(672)
db = lambda x: 10 * np.log10(np.maximum(x, 1e-6))
OUT = HERE / "figures_upchannelization"; OUT.mkdir(exist_ok=True)


def figure(ant):
    P = np.asarray(D[ant])
    G = P.reshape(len(P), 672, 32).sum(-1) / 32            # coarse channels, same unit per fine channel

    fig, ax = plt.subplots(3, 2, figsize=(16, 15))
    # (a)/(b) whole band, 4 s (32 ms per pixel for plotting)
    for j, (X, ff, name) in enumerate(((G, fg, "Before: 672 channels of 0.3 MHz"), (P, f, "After: 21 504 channels of 9.375 kHz"))):
        w = db(X.reshape(128, 32, -1).mean(1))
        im = ax[0, j].imshow(w.T, aspect="auto", origin="lower", cmap="viridis", extent=[0, 4.096, ff[0], ff[-1]],
                             vmin=np.percentile(w, 1), vmax=np.percentile(w, 99.5))
        fig.colorbar(im, ax=ax[0, j], label="dB"); ax[0, j].set(title=f"{name} · whole band (32 ms per pixel)", xlabel="Time (s)", ylabel="Frequency (MHz)")
    # (c)/(d) zoom 448–458 MHz, first 1.5 s, native 1 ms resolution
    lo, hi, t1 = 448.0, 458.0, 1500
    sg = (fg >= lo) & (fg <= hi); sf = (f >= lo - 0.15) & (f <= hi + 0.15)
    for j, (X, ff, s, name) in enumerate(((G, fg, sg, f"Before: {sg.sum()} channels"), (P, f, sf, f"After: {sf.sum()} channels (max. of 3 per row)"))):
        w = db(X[:t1, s])
        if j == 1:                                     # 1099 rows in ~380 pixels: maximum of every 3 fine channels
            n3 = w.shape[1] // 3 * 3                   # so that single-channel lines do not disappear
            w = w[:, :n3].reshape(len(w), -1, 3).max(-1)
        im = ax[1, j].imshow(w.T, aspect="auto", origin="lower", cmap="viridis", extent=[0, t1 / 1000, ff[s][0], ff[s][-1]],
                             vmin=np.percentile(w, 1), vmax=np.percentile(w, 99.7), interpolation="nearest")
        fig.colorbar(im, ax=ax[1, j], label="dB"); ax[1, j].set(title=f"{name} · zoom {lo:.0f}–{hi:.0f} MHz, 1 ms per pixel", xlabel="Time (s)", ylabel="Frequency (MHz)")
    # (e) mean spectrum 405–432 MHz
    lo, hi = 405.0, 432.0
    s = (f >= lo) & (f <= hi); s2 = (fg >= lo) & (fg <= hi)
    ax[2, 0].plot(f[s], db(P[:, s].mean(0)), lw=0.6, color=plt.cm.viridis(0.35), label="after (9.375 kHz)")
    ax[2, 0].step(fg[s2], db(G[:, s2].mean(0)), where="mid", lw=1.8, color="k", label="before (0.3 MHz)")
    ax[2, 0].set(title=f"Mean spectrum over 4.1 s, {lo:.0f}–{hi:.0f} MHz", xlabel="Frequency (MHz)", ylabel="dB"); ax[2, 0].legend()
    # (f) 6 coarse channels around a narrow line (400.6 MHz): resolved lines and scalloping
    lo, hi = 399.6, 401.4
    s = (f >= lo) & (f <= hi); s2 = (fg >= lo - 0.1) & (fg <= hi + 0.1)
    ax[2, 1].plot(f[s], db(P[:, s].mean(0)), ".-", ms=3, lw=0.8, color=plt.cm.viridis(0.35), label="after: 32 channels per coarse channel")
    ax[2, 1].step(fg[s2], db(G[:, s2].mean(0)), where="mid", lw=1.8, color="k", label="before")
    for c in fg[s2]:
        ax[2, 1].axvline(c - 0.15, color="0.8", lw=0.8, zorder=0)
    ax[2, 1].set(title="Zoom on 6 coarse channels (grey: edges); the PFB scalloping is visible", xlabel="Frequency (MHz)", ylabel="dB"); ax[2, 1].legend(fontsize=10)
    fig.suptitle(f"Antenna {ant} (input {ant}) · site baseband 2026-09-02, before and after ×32 channelization · both integrated to 1 ms", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / f"antenna{ant}.png", dpi=100); plt.close(fig)
    print(OUT / f"antenna{ant}.png", flush=True)


for ant in ([int(x) for x in sys.argv[1:]] or range(D.shape[0])):
    figure(ant)
