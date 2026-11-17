#!/usr/bin/env python3
"""Where FRBs are lost in the 64-beam test: recall versus injected S/N, width, scattering, duration and DM.

Sources (from evaluate_test_64b.py, without matching events again):
  RUN/assignment.csv          parameters of each injected file (one row per file, same order)
  RUN/best_snr_per_file.npy   best S/N of an event that hits each file (-inf if none):
                              same beam, DM ±10 %, |Δt| ≤ 50 ms (NEG: any DM within ±100 ms)
(Runs of the earlier Spanish-named chain, with asignacion.csv and mejor_snr_por_archivo.npy, are also read.)

Definitions
  detected         best S/N ≥ THRESHOLD (6.54 = 10 noise FP per hour in that run, see results.md)
  universe         positives A, B and cut (P) with DM ≤ 1000, as in results.md
  injected S/N     A and B: snr_esperado (optimal boxcar, full band); P: snr_visible_esperado (visible part)
  duration         optimal boxcar of the dedispersed burst (boxcar_opt_ms; P: boxcar_opt_visible_ms):
                   intrinsic width + scattering + smearing; pirate tries boxcars up to 32 ms
  scattering       scat_ref_s, at 400.2 MHz (at 300 MHz it is 3.2 times larger)
  non-FRB FP       % of NEG files (same equivalent DM, width and scattering as their pair) with an event ≥ threshold
  probable cause   (in order) injected S/N < threshold → "weak"; there is an event ≥ threshold in the time window but
                   at a DM outside ±10 % → "DM shifted"; there is an event at the right DM but with S/N < threshold →
                   "below threshold"; otherwise → "no nearby event" (from nearby_events_64b.py, pirate environment)

Usage (an environment with pandas, e.g. the simulator one): python loss_analysis_64b.py RUN [--threshold 6.54 --fp-h 8.9]
Output in RUN/analysis_by_parameter/: table_per_file.csv, recall_per_bin.csv, causes.csv,
recall_per_parameter.png, measured_snr_vs_duration.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--threshold", type=float, default=6.54)
ap.add_argument("--dm-max", type=float, default=1000.0)
ap.add_argument("--fp-h", default="8.9", help="noise FP per hour at that threshold (from results.md), only for the title")
a = ap.parse_args()
OUT = a.run / "analysis_by_parameter"
OUT.mkdir(exist_ok=True)
U, BOX_MAX = a.threshold, 32.0


def first_existing(*names):
    for n in names:
        if (a.run / n).exists():
            return a.run / n
    raise SystemExit(f"none of {names} in {a.run}")


d = pd.read_csv(first_existing("assignment.csv", "asignacion.csv"))
d["best"] = np.load(first_existing("best_snr_per_file.npy", "mejor_snr_por_archivo.npy"))
assert len(d) == len(d["best"]), "assignment.csv and best_snr_per_file.npy do not have the same length"
d["det"] = d["best"] >= U
d["dm_v"] = np.where(d["set"] == "NEG", d["dm_equivalente_pos"], d["dm"])
d["snr_inj"] = np.where(d["set"] == "P", d["snr_visible_esperado"], d["snr_esperado"])
d["dur_ms"] = np.where(d["set"] == "P", d["boxcar_opt_visible_ms"], d["boxcar_opt_ms"])
d["width_ms"] = d["width_s"] * 1e3
d["scat_ms"] = d["scat_ref_s"] * 1e3
# NEG: duration of the positive pair (same width, scattering and equivalent DM), if it exists
pair = d.set_index("name")["dur_ms"]
m_neg = (d["set"] == "NEG") & d["par_de"].notna()
d.loc[m_neg, "dur_ms"] = d.loc[m_neg, "par_de"].map(pair).values

u = d[d["dm_v"] <= a.dm_max].copy()
pos = u[u["set"].isin(["A", "B", "P"])].copy()
neg = u[u["set"] == "NEG"]
CAUSES = ["weak: injected S/N < threshold", "DM shifted > 10 % (event ≥ threshold at another DM)",
          "right DM, measured S/N 5–6.5 (below threshold)", "no event at its DM (not even below threshold)"]
ne = OUT / "nearby_events.csv"
if not ne.exists():
    raise SystemExit(f"{ne} is missing: first run (pirate environment) python nearby_events_64b.py {a.run}")
pos = pos.merge(pd.read_csv(ne), on="name", how="left")
pos["cause"] = np.select([pos["snr_inj"] < U, pos["snr_other_dm"] >= U, pos["snr_dm_ok"].notna()], CAUSES[:3], CAUSES[3])
pos.loc[pos["det"], "cause"] = "detected"
u.to_csv(OUT / "table_per_file.csv", index=False)
print(f"source: {a.run}  threshold {U}  DM ≤ {a.dm_max:.0f}: " + ", ".join(f"{s} {int((u.set == s).sum())}" for s in ("A", "B", "P", "NEG")))

NAME = {"A": "A · S/N 8", "B": "B · fixed flux", "P": "cut"}
COL = {"A": plt.cm.viridis(0.15), "B": plt.cm.viridis(0.5), "P": plt.cm.viridis(0.82)}
MARK = plt.cm.viridis(0.95)
BINS = {
    "snr_inj": ([0, 4, 6.54, 8.01, 12, 20, 40, 1e4], "injected S/N (optimal boxcar)", ["<4", "4–6.5", "6.5–8", "8–12", "12–20", "20–40", ">40"]),
    "dur_ms": ([0, 2, 4, 8, 16, 32, 64, 1e5], "effective burst duration (ms)", ["≤2", "2–4", "4–8", "8–16", "16–32", "32–64", ">64"]),
    "width_ms": ([0, 0.3, 1, 3, 10, 100], "intrinsic width (ms)", ["<0.3", "0.3–1", "1–3", "3–10", ">10"]),
    "scat_ms": ([-1, 0.01, 0.1, 1, 10, 1e5], "scattering at 400 MHz (ms)", ["<0.01", "0.01–0.1", "0.1–1", "1–10", ">10"]),
    "dm_v": ([0, 250, 500, 750, 1000.01], "DM (pc cm⁻³)", ["≤250", "250–500", "500–750", "750–1000"]),
}

rows = []
for var, (b, _, labels) in BINS.items():
    for s in ("A", "B", "P", "NEG"):
        g = u[u["set"] == s]
        if g[var].isna().all():
            continue
        k = pd.cut(g[var], b, labels=labels, right=False)
        for e in labels:
            gg = g[k == e]
            rows.append(dict(variable=var, bin=e, set=s, n=len(gg), detected=int(gg["det"].sum()),
                             pct=round(100 * gg["det"].mean(), 1) if len(gg) else np.nan))
rb = pd.DataFrame(rows)
rb.to_csv(OUT / "recall_per_bin.csv", index=False)

# Figure 1: recall per parameter (one panel per variable), NEG in grey = non-FRB FP of the same bin
fig, axs = plt.subplots(2, 3, figsize=(17, 9.5))
for ax, (var, (b, xl, labels)) in zip(axs.flat, BINS.items()):
    x = np.arange(len(labels))
    sets = [s for s in ("A", "B", "P") if not rb[(rb.variable == var) & (rb.set == s)].empty]
    w = 0.8 / (len(sets) + 1)
    for j, s in enumerate(sets + ["NEG"]):
        r = rb[(rb.variable == var) & (rb.set == s)].set_index("bin").reindex(labels)
        if r["n"].isna().all():
            continue
        xs = x + (j - len(sets) / 2) * w
        y = r["pct"].values.astype(float)
        c = "0.6" if s == "NEG" else COL[s]
        ax.bar(xs, np.nan_to_num(y), w * 0.92, color=c, label=("NEG: non-FRB FP" if s == "NEG" else NAME[s]),
               hatch="//" if s == "NEG" else None, edgecolor="white" if s != "NEG" else "0.4", linewidth=0.5)
        for xi, yi, ni in zip(xs, y, r["n"].values):
            if ni and ni > 0:
                ax.text(xi, (0 if np.isnan(yi) else yi) + 1.5, f"{int(ni)}", ha="center", va="bottom", fontsize=7, color="0.35", rotation=90)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 112)
    ax.set_xlabel(xl)
    ax.set_ylabel("% with event ≥ threshold")
    ax.grid(axis="y", alpha=0.3)
    if var == "dur_ms":
        ax.axvline(4.5, color="0.2", ls="--", lw=1.2)
        ax.text(4.55, 104, "pirate max.\nboxcar (32 ms)", color="0.2", fontsize=9, va="top")
    if var == "snr_inj":
        ax.axvline(1.5, color="0.2", ls="--", lw=1.2)
        ax.text(1.55, 104, f"threshold {U}", color="0.2", fontsize=9, va="top")
from matplotlib.patches import Patch
h0, _ = axs.flat[0].get_legend_handles_labels()
axs.flat[0].legend(handles=h0 + [Patch(fc="0.6", ec="0.4", hatch="//", label="NEG: non-FRB FP (other panels)")], loc="upper left", fontsize=9, bbox_to_anchor=(0, 0.92))

# last panel: loss causes per set
ax = axs.flat[5]
cc = dict(zip(CAUSES, [plt.cm.viridis(0.95), plt.cm.viridis(0.6), plt.cm.viridis(0.35), "0.45"]))
tab = pos.groupby(["set", "cause"]).size().unstack(fill_value=0).reindex(["A", "B", "P"])
left = np.zeros(3)
for c in CAUSES:
    v = tab.get(c, pd.Series(0, index=tab.index)).values
    ax.barh(np.arange(3), v, left=left, color=cc[c], label=c, edgecolor="white")
    for i, (vi, li) in enumerate(zip(v, left)):
        if vi >= 5:
            ax.text(li + vi / 2, i, str(vi), ha="center", va="center", color="black" if c == CAUSES[0] else "white", fontsize=10)
    left += v
for i, s in enumerate(["A", "B", "P"]):
    n, det = int((pos.set == s).sum()), int(pos[pos.set == s]["det"].sum())
    ax.text(left[i] + 1, i, f"{n - det} lost of {n}", va="center", fontsize=9)
ax.set_yticks(range(3), [NAME[s] for s in ["A", "B", "P"]])
ax.invert_yaxis()
ax.set_xlim(0, max(left) * 1.6 + 5)
ax.set_xlabel("lost FRBs (no event ≥ threshold)")
ax.set_title("Probable cause of each loss", fontsize=11, loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.5, frameon=False, ncol=1)
fig.suptitle(f"Real-time 64-beam test (group 3, no RFI, DM ≤ {a.dm_max:.0f}): % of files with an event, S/N threshold {U} "
             f"(= {a.fp_h} noise FP per hour) · number = files in the bin", fontsize=12.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "recall_per_parameter.png", dpi=110, facecolor="white")
plt.close(fig)

# Figure 2: measured / injected S/N against duration; lost files in a separate band
fig, ax = plt.subplots(figsize=(11, 7.5))
for s in ("A", "B", "P"):
    g = pos[pos.set == s]
    dt = g[g.det]
    ax.scatter(dt["dur_ms"], dt["best"] / dt["snr_inj"], s=16, color=COL[s], alpha=0.75, label=f"{NAME[s]}: detected ({len(dt)})")
    lost = g[~g.det]
    yj = 0.03 * np.exp(np.random.default_rng(1).uniform(-0.25, 0.25, len(lost)))
    ax.scatter(lost["dur_ms"], yj, s=22, marker="x", color=COL[s], alpha=0.9, label=f"{NAME[s]}: lost ({len(lost)})")
ax.axvline(BOX_MAX, color="0.2", ls="--")
ax.text(BOX_MAX * 1.05, 2.2, "pirate maximum\nboxcar (32 ms)", color="0.2", fontsize=10, va="top")
ax.axhline(1, color="0.4", lw=0.8)
ax.axhspan(0.02, 0.045, color="0.92", zorder=0)
ax.text(1.05, 0.047, "lost (no event ≥ threshold; no S/N value)", fontsize=9, color="0.35")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(0.02, 2.5)
ax.set_xlabel("effective burst duration (optimal boxcar, ms)")
ax.set_ylabel("S/N measured by pirate / injected S/N")
ax.set_title(f"How much S/N pirate recovers versus burst duration (DM ≤ {a.dm_max:.0f})", loc="left")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.11), frameon=False)
fig.tight_layout()
fig.savefig(OUT / "measured_snr_vs_duration.png", dpi=110, facecolor="white")
plt.close(fig)

# Console summaries
c = pos.groupby(["set", "cause"]).size().unstack(fill_value=0)
c.to_csv(OUT / "causes.csv")
print("\nProbable cause per set (DM ≤ %.0f):\n%s" % (a.dm_max, c.to_string()))
print(f"\nRecall: " + ", ".join(f"{s} {pos[pos.set == s]['det'].mean() * 100:.1f} % ({int(pos[pos.set == s]['det'].sum())}/{int((pos.set == s).sum())})" for s in ("A", "B", "P")))
free = pos["det"] | (pos["snr_other_dm"] >= U)
print("Recall without requiring DM ±10 %: " + ", ".join(f"{s} {free[pos.set == s].mean() * 100:.1f} %" for s in ("A", "B", "P")))
print(f"Non-FRB FP (NEG): {neg['det'].mean() * 100:.1f} % ({int(neg['det'].sum())}/{len(neg)})")
dt = pos[pos.det]
for lo, hi in ((0, 8), (8, 32), (32, 1e9)):
    m = dt[(dt.dur_ms > lo) & (dt.dur_ms <= hi)]
    if len(m):
        r = m["best"] / m["snr_inj"]
        print(f"Measured/injected S/N, duration {lo}–{hi:.0f} ms: median {r.median():.2f}  p10 {r.quantile(.1):.2f}  p90 {r.quantile(.9):.2f}  (n {len(m)})")
print("\n" + rb.pivot_table(index=["variable", "bin"], columns="set", values=["n", "pct"], sort=False).to_string())
print("\noutput:", OUT)
