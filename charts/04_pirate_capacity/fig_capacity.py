#!/usr/bin/env python3
"""How pirate's memory and capacity scale with the maximum DM, the block size and max_gpu_clag (64 beams, 21 504 channels).

Data: timings.json (timings.py, 2026-10-21): `pirate_frb show dedisperser` (tree state in GPU/RAM, PCIe at real time)
and `pirate_frb time_dedisperser -H -n 24` (reserved memory and real-time beams with the GPU at 100 %), RTX 4090.
Ranks 13 and 14 have no compiled kernel in pirate: memory only (sweep_show.json, from sweep.py).
Output: pirate_capacity.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "timings.json").read_text())
S = json.loads((HERE / "sweep_show.json").read_text())
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "legend.fontsize": 11})
CG, CR = plt.cm.viridis(0.25), plt.cm.viridis(0.75)


def find(kind, rank, nt, clag, ntrees):
    return next(r for r in R if (r["kind"], r["rank"], r["nt"], r["clag"], r["ntrees"]) == (kind, rank, nt, clag, ntrees))


fig, axs = plt.subplots(2, 2, figsize=(16, 9))

# (a) tree memory versus maximum DM
r14 = next(s for s in S if s["rank"] == 14 and s["nt"] == 2048)
cases = [("552\nr14", dict(ring_gpu=r14["gpu"], ring_host=r14["host"], beams_rt=None, h2g_rt=r14["h2g"])),
         ("1105\nr15 (test)", find("dm_nt", 15, 2048, 5, 1)),
         ("2209\nr15 +1", find("trees", 15, 2048, 5, 2)),
         ("2209\nr16", find("dm_nt", 16, 2048, 5, 1)),
         ("4418\nr15 +2", find("trees", 15, 2048, 5, 3)),
         ("4418\nr16 +1", find("trees", 16, 2048, 1, 2))]
x = np.arange(len(cases))
XL = "Maximum DM · r = rank (1 ms tree) · +n = extra trees (2 and 4 ms)"
ax = axs[0, 0]
g = np.array([c[1]["ring_gpu"] for c in cases]); h = np.array([c[1]["ring_host"] for c in cases])
ax.bar(x, g, color=CG, label="on GPU")
ax.bar(x, h, bottom=g, color=CR, label="in RAM")
for i, (gi, hi) in enumerate(zip(g, h)):
    ax.text(i, gi + hi + 4, f"{gi + hi:.0f}", ha="center", fontsize=12)
ax.set_xticks(x, [c[0] for c in cases], fontsize=11)
ax.set_xlabel(XL, fontsize=11.5)
ax.set_ylabel("Tree state (GiB)")
ax.set_ylim(0, 290)
ax.set_title("(a) Tree memory versus maximum DM")
ax.legend(loc="upper left")
ax.text(0.27, 0.95, "rank +1 (DM ×2) → memory ×4\nextra tree → much cheaper", transform=ax.transAxes,
        va="top", fontsize=12, bbox=dict(fc="white", ec="0.8"))

# (b) capacity versus DM
ax = axs[0, 1]
for i, c in enumerate(cases):
    bi = c[1]["beams_rt"]
    ax.bar(i, bi or 0, color=plt.cm.viridis(0.45))
    ax.text(i, (bi or 0) + (15 if bi else 90), (f"{bi:.0f}\n" if bi else "no compiled\nkernel\n") + f"PCIe {c[1]['h2g_rt']:.1f} GB/s",
            ha="center", va="bottom", fontsize=10.5)
ax.axhline(64, color="k", ls="--", lw=1.5)
ax.text(5.45, 50, "64 beams", ha="right", va="top", fontsize=11)
ax.set_xticks(x, [c[0] for c in cases], fontsize=11)
ax.set_xlabel(XL, fontsize=11.5)
ax.set_ylabel("Real-time beams (GPU at 100 %)")
ax.set_ylim(0, 950)
ax.set_title("(b) Capacity of one RTX 4090  ·  PCIe RAM→GPU with 64 beams")

# (c) block size
ax = axs[1, 0]
nts = [512, 1024, 2048, 4096, 8192]
rr = [find("dm_nt", 15, nt, max(1, round(10240 / nt)), 1) for nt in nts]
ax.plot(nts, [r["beams_rt"] for r in rr], "o-", color=CG, lw=2, label="DM 1105 (GPU 13–21 GB, RAM 22–24 GB)")
for r in rr:
    ax.annotate(f"GPU {r['gpu_GB']:.0f} · RAM {r['cpu_GB']:.0f}", (r["nt"], r["beams_rt"]), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=10, color="0.25")
r16 = [find("dm_nt", 16, nt, max(1, round(10240 / nt)), 1) for nt in nts]
ok = [r for r in r16 if "beams_rt" in r]
ax.plot([r["nt"] for r in ok], [r["beams_rt"] for r in ok], "s-", color=CR, lw=2,
        label="DM 2209 (GPU 20–25 GB, RAM ~120 GB)")
ax.plot([8192], [0], "x", color=CR, ms=12, mew=3, clip_on=False)
ax.text(8192, 40, "DM 2209: does not fit\n(GPU 35 GB)", ha="center", fontsize=10, color="0.25")
ax.set_xscale("log", base=2)
ax.set_xticks(nts, [f"{n}\n({n / 1000:g} s)" for n in nts])
ax.set_xlabel("Block size (1 ms samples) = minimum wait before processing")
ax.set_ylabel("Real-time beams")
ax.set_ylim(0, 1000)
ax.set_title("(c) Block size: little effect on capacity")
ax.legend(loc="center", bbox_to_anchor=(0.5, 0.45))

# (d) max_gpu_clag
ax = axs[1, 1]
cl = [r for r in R if r["kind"] == "clag"]
ok = [r for r in cl if "beams_rt" in r]
ax.plot([r["clag"] for r in ok], [r["beams_rt"] for r in ok], "o-", color=plt.cm.viridis(0.45), lw=2)
for r in ok:
    if r["clag"] >= 5:
        ax.annotate(f"GPU {r['gpu_GB']:.0f}\nRAM {r['cpu_GB']:.0f}", (r["clag"], r["beams_rt"]), textcoords="offset points",
                    xytext=(-26, 4), ha="center", fontsize=10, color="0.25")
ax.text(7.0, 620, "\n".join(f"clag {r['clag']}: GPU {r['gpu_GB']:.0f} · RAM {r['cpu_GB']:.0f} GB" for r in ok if r["clag"] < 5),
        fontsize=10.5, color="0.25", va="top", bbox=dict(fc="white", ec="0.8"))
for r in cl:
    if "beams_rt" not in r:
        ax.plot([r["clag"]], [0], "x", color="k", ms=12, mew=3, clip_on=False)
        ax.text(r["clag"], 60, f"does not fit\nGPU {r['gpu_GB']:.0f} GB", ha="center", fontsize=10, color="0.25")
ax.axvline(5, color="0.6", ls=":")
ax.text(5.2, 300, "test", fontsize=11, color="0.4")
ax.set_xlabel("max_gpu_clag (2.048 s blocks the tree keeps on the GPU), DM 1105")
ax.set_ylabel("Real-time beams")
ax.set_xlim(0, 17)
ax.set_ylim(0, 1350)
ax.set_xticks([1, 2, 3, 5, 8, 12, 16])
ax.set_title("(d) More state on the GPU → less PCIe → more beams")

fig.suptitle("pirate resources: 64 beams × 21 504 channels, RTX 4090 (measured with time_dedisperser)", fontsize=17)
fig.text(0.5, 0.01, "GPU memory and RAM in GB = what pirate reserves for dedispersion (does not include the input "
         "ring buffer).\nAt full speed, in every configuration the RAM→GPU copy runs at ~20–23 GB/s: PCIe is what "
         "limits the capacity when the tree uses RAM.\nRank 17 or higher is not allowed in pirate (maximum 16); for higher "
         "DM, trees with 2 and 4 ms samples are added.", ha="center", va="bottom", fontsize=11.5, color="0.3")
fig.tight_layout(rect=(0, 0.085, 1, 0.96))
out = HERE / "pirate_capacity.png"
fig.savefig(out, dpi=150)
print(out)
