#!/usr/bin/env python3
"""Figure of quantization_21504.csv with the theoretical efficiency of each variant (quantization_theory.py)."""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
(HERE / "plots_21504").mkdir(exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rows = list(csv.DictReader(open(HERE / "plots_21504" / "quantization_21504.csv")))
names = ["8 bits", "4b 2lev fixed", "4b 3lev fixed", "4b 1lev fixed", "4b pirate"]
theory = {"8 bits": 0.9974, "4b 2lev fixed": 0.9822, "4b 3lev fixed": 0.9903, "4b 1lev fixed": 0.9535, "4b pirate": 0.9822}
fig, ax = plt.subplots(figsize=(11, 6))
for k, name in enumerate(names):
    for j, s in enumerate(("A", "B")):
        sub = [r for r in rows if r["set"] == s and float(r["float"]) > 4]
        rel = np.array([float(r[name]) / float(r["float"]) for r in sub])
        snr = np.array([float(r["float"]) for r in sub])
        xs = k + (j - 0.5) * 0.28 + np.random.default_rng(k + 10 * j).uniform(-0.05, 0.05, rel.size)
        sc = ax.scatter(xs, 100 * (rel - 1), c=np.log10(snr), cmap="viridis", vmin=0.6, vmax=2.2,
                        marker="o" if s == "A" else "s", s=34, edgecolor="k", lw=0.3,
                        label=f"set {s}" if k == 0 else None)
    ax.hlines(100 * (theory[name] - 1), k - 0.35, k + 0.35, color="C3", lw=2.5,
              label="theory (weak signal)" if k == 0 else None)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(names)), ["8 bits\n(the .fil)", "4 bits\n2 levels/σ", "4 bits\n3 levels/σ",
                                  "4 bits\n1 level/σ", "4 bits pirate\nscale/offset per 256 samples"])
ax.set_ylabel("S/N relative to the unquantized data (%)")
ax.set_title("Quantization: same noise and signal realization (group-3 test, 21 504 channels)\n"
             "colour = burst S/N; the high-S/N ones measure the loss more precisely")
cb = fig.colorbar(sc, ax=ax)
cb.set_label("log10 S/N (unquantized)")
ax.set_ylim(-24, 7)
ax.annotate("A_007: scattering 25.6 ms;\nthe per-block offset of 256 ms\nsubtracts part of the pulse",
            xy=(4 - 0.14, -21.7), xytext=(2.6, -20), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.6))
ax.legend(loc="lower left", fontsize=9)
fig.savefig(HERE / "plots_21504" / "quantization_21504_theory.png", dpi=110, bbox_inches="tight")
print(HERE / "plots_21504" / "quantization_21504_theory.png")
