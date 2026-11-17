#!/usr/bin/env python
"""T3b · Which FRBs are missed? Diagnosis of each A/B/cut file (DM ≤ 1000) not detected at threshold 6.5.

Categories (in order):
  low_expected        expected S/N < threshold: the simulator already leaves it below the threshold
  out_of_tolerance    there is an event ≥ threshold in the same beam within ±1 s of t_top, but DM or time fails
  measured_low        expected S/N ≥ threshold, but the best matching event is below the threshold (or there is none)

Usage (pirate environment): python missed.py [RUN_FOLDER]   (or export CHARTS_RUN=...)
Output: RUN/analysis/ (or $CHARTS_ANALYSIS_OUT): missed.md, missed.json, 5_missed.png
"""
import csv
import json
import re
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = Path(__file__).resolve().parent
if len(sys.argv) < 2 and "CHARTS_RUN" not in os.environ:
    sys.exit(f"Usage: python {sys.argv[0]} RUN_FOLDER   (or export CHARTS_RUN=...)")
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ["CHARTS_RUN"])
OUTPUT = Path(os.environ.get("CHARTS_ANALYSIS_OUT", RUN / "analysis"))   # tables and figures
OUTPUT.mkdir(parents=True, exist_ok=True)
DD = Path(__file__).resolve().parents[2] / "06_configs" / "charts_64b_dm1000.yml"
SEQ, DM_MIN, TOL_DM, TOL_T, THRESHOLD = 200, 20.0, 0.10, 0.05, 6.50
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})


def first_existing(*names):
    for n in names:
        if (RUN / n).exists():
            return RUN / n
    sys.exit(f"ERROR: none of {names} in {RUN}")


def delay_edge_to_top():
    import pirate_frb as pf
    cfg = pf.DedispersionConfig.from_yaml(str(DD))
    f_lo, f_hi = cfg.zone_freq_edges[0], cfg.zone_freq_edges[-1]
    return 1e-3 / cfg.dm_per_unit_delay() * (f_lo ** -2 - 501.6 ** -2) / (f_lo ** -2 - f_hi ** -2)


K_TOP = delay_edge_to_top()
ev_re = re.compile(r"beam_id=(\d+), dm=([\d.e+-]+), fpga_timestamp=(-?\d+), width=([\d.e+-]+) ms.*snr=([\d.e+-]+)")
ev = np.array([(int(m.group(1)), float(m.group(2)), int(m.group(3)) / SEQ * 1e-3, float(m.group(4)), float(m.group(5)))
               for m in map(ev_re.search, (RUN / "sifter.log").read_text(errors="replace").splitlines()) if m])
ev = ev[ev[:, 1] >= DM_MIN]
e_b, e_dm, e_w, e_s = ev[:, 0].astype(int), ev[:, 1], ev[:, 3], ev[:, 4]
e_t = ev[:, 2] - e_dm * K_TOP


def f(r, k, d=np.nan):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return d


rows = [r for r in csv.DictReader(open(first_existing("assignment.csv", "asignacion.csv")))
        if r["set"] != "NEG" and float(r["dm"]) <= 1000]
out = []
for r in rows:
    b, dm = int(r["beam"]), float(r["dm"])
    t_abs = int(r.get("start_sample") or r["inicio_muestra"]) / 1000 + float(r["t_top_s"])
    box = (f(r, "boxcar_opt_ms", 0) or f(r, "boxcar_opt_visible_ms", 0)) / 1000
    exp = f(r, "snr_visible_esperado") if r["set"] == "P" else f(r, "snr_esperado")
    tol_t = max(TOL_T, box)
    sel = e_b == b
    ok = sel & (np.abs(e_dm - dm) <= max(TOL_DM * dm, 1.0)) & (np.abs(e_t - t_abs) <= tol_t)
    best = e_s[ok].max() if ok.any() else 0.0
    if best >= THRESHOLD:
        continue
    near = sel & (np.abs(e_t - t_abs) <= 1.0) & (e_s >= THRESHOLD)
    d = dict(name=r["name"], set=r["set"], edge=r.get("borde", ""), dm=dm, snr_exp=exp, best=best,
             width_ms=f(r, "width_s") * 1e3, scat_600_ms=f(r, "scat_600_s") * 1e3, boxcar_opt_ms=box * 1e3,
             frac_visible=f(r, "frac_energia_visible"), n_channels=f(r, "n_canales_emision"))
    if np.isfinite(exp) and exp < THRESHOLD:
        d["cat"] = "low_expected"
    elif near.any():
        k = np.flatnonzero(near)[np.argmax(e_s[near])]
        d.update(cat="out_of_tolerance", ev_snr=e_s[k], ev_dm=e_dm[k], ev_w=e_w[k],
                 err_dm=e_dm[k] - dm, err_t_ms=(e_t[k] - t_abs) * 1e3, tol_t_ms=tol_t * 1e3)
    else:
        d["cat"] = "measured_low"
    out.append(d)

tot = {s: sum(r["set"] == s for r in rows) for s in ("A", "B", "P")}
cats = ("low_expected", "measured_low", "out_of_tolerance")
txt = f"# T3b · FRBs missed at threshold {THRESHOLD} (DM ≤ 1000)\n\n"
txt += "| Set | total | missed | " + " | ".join(cats) + " |\n|---|---|---|---|---|---|\n"
for s in ("A", "B", "P"):
    p = [d for d in out if d["set"] == s]
    txt += f"| {s} | {tot[s]} | {len(p)} | " + " | ".join(str(sum(d['cat'] == c for d in p)) for c in cats) + " |\n"
txt += "\n## Cut files missed, by edge (inicio = start, fin = end)\n\n"
for bo in ("inicio", "fin"):
    p = [d for d in out if d["set"] == "P" and d["edge"] == bo]
    txt += f"- {bo}: {len(p)} missed of {sum(r['set'] == 'P' and r.get('borde') == bo for r in rows)}; " \
           + ", ".join(f"{c} {sum(d['cat'] == c for d in p)}" for c in cats) + "\n"
txt += ("\n## measured_low and out_of_tolerance, one by one\n\n| file | exp. S/N | best | DM | width ms | scat600 ms | "
        "boxcar ms | frac vis | category | detail |\n|---|---|---|---|---|---|---|---|---|---|\n")
for d in sorted([d for d in out if d["cat"] != "low_expected"], key=lambda d: -(d["snr_exp"] if np.isfinite(d["snr_exp"]) else 0)):
    det = (f"event S/N {d['ev_snr']:.1f} DM {d['ev_dm']:.0f} (ΔDM {d['err_dm']:+.0f}), Δt {d['err_t_ms']:+.0f} ms "
           f"(tol {d['tol_t_ms']:.0f}), width {d['ev_w']:g}") if d["cat"] == "out_of_tolerance" else ""
    txt += (f"| {d['name']} | {d['snr_exp']:.1f} | {d['best']:.2f} | {d['dm']:.0f} | {d['width_ms']:.2f} | "
            f"{d['scat_600_ms']:.1f} | {d['boxcar_opt_ms']:.0f} | {d['frac_visible']:.2f} | {d['cat']} | {det} |\n")
txt += "\n## Set A (S/N 8) by optimal boxcar (intrinsic width + scattering)\n\n| optimal boxcar | n | recall |\n|---|---|---|\n"
missed_A = {d["name"] for d in out if d["set"] == "A"}
bxA = [((f(r, "boxcar_opt_ms", 0)), r["name"] in missed_A) for r in rows if r["set"] == "A"]
for lo, hi in ((0, 8), (8, 50), (50, 200), (200, 1e9)):
    sel_ = [p_ for b_, p_ in bxA if lo <= b_ < hi]
    if sel_:
        txt += f"| {lo:g}–{hi:g} ms | {len(sel_)} | {100 * (1 - np.mean(sel_)):.1f} % |\n".replace("–1e+09", "+")
(OUTPUT / "missed.md").write_text(txt)
(OUTPUT / "missed.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))
print(txt)

# figure: expected S/N of the missed files per category
fig, ax = plt.subplots(figsize=(8, 4.2))
col = {"low_expected": plt.cm.viridis(0.0), "measured_low": plt.cm.viridis(0.5), "out_of_tolerance": plt.cm.viridis(0.95)}
bins = np.logspace(-1.5, 2.5, 40)
ax.hist([[d["snr_exp"] for d in out if d["cat"] == c and np.isfinite(d["snr_exp"])] for c in cats], bins=bins,
        stacked=True, color=[col[c] for c in cats], edgecolor="0.3", lw=0.3,
        label=["expected S/N below the threshold", "expected ≥ threshold, measured below", "detected out of tolerance"])
ax.axvline(THRESHOLD, color="black", ls="--", lw=1)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("expected S/N (cut files: visible S/N)")
ax.set_ylabel("missed FRBs")
ax.set_title(f"FRBs missed at threshold {THRESHOLD} (A + B + cut, DM ≤ 1000): {len(out)}")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT / "5_missed.png", dpi=130)
