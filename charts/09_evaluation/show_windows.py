#!/usr/bin/env python3
"""Shows windows of the DM-time map saved by the grouper (windows/ folder): 6 FRB, 3 non-FRB and 3 noise.
Each window: 256 DM cells (centred on the event) × 384 time cells (previous chunk, chunk of the event and next one).
Needs windows/index.csv with the columns npz, k, name, label (frb | nofrb | noise), steady, dm, snr, width_ms.
Usage: python show_windows.py RUN_FOLDER OUTPUT.png"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

run, out = Path(sys.argv[1]), sys.argv[2]
idx = pd.read_csv(run / "windows" / "index.csv")
sel = pd.concat([idx[idx.label == "frb"].sort_values("snr", ascending=False).head(6),
                 idx[idx.label == "nofrb"].sort_values("snr", ascending=False).head(3),
                 idx[(idx.label == "noise") & (idx.steady == 1)].sample(3, random_state=0)])
fig, axs = plt.subplots(3, 4, figsize=(16, 9))
for ax, (_, r) in zip(axs.flat, sel.iterrows()):
    z = np.load(run / "windows" / r.npz)
    v = z["window"][r.k].astype(np.float32)
    nt = v.shape[1] // 3
    im = ax.imshow(v, aspect="auto", cmap="viridis", vmin=0, vmax=max(6, np.percentile(v, 99.9)), origin="lower")
    ax.axvline(nt, color="white", lw=0.6, ls=":")
    ax.axvline(2 * nt, color="white", lw=0.6, ls=":")
    ax.set_title(f"{r.label} · {r['name']}\nDM {r.dm:.0f} · S/N {r.snr:.1f} · {r.width_ms:.0f} ms", fontsize=9)
    ax.set_xlabel("time cell (16 ms)", fontsize=8)
    ax.set_ylabel("DM cell (0.54) − 128", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, label="S/N")
fig.suptitle("Windows of the DM-time map (out_max) saved by the grouper · native resolution", fontsize=12)
fig.tight_layout()
fig.savefig(out, dpi=110, facecolor="white")
