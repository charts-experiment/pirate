#!/usr/bin/env python
"""T3e · File-by-file comparison: subbands [0,0,1,2,1] against full band, at equal noise FP.

For each file (A/B/P/NEG, DM ≤ 1000) it takes the best matching event (same criteria as evaluate_test_64b.py)
in each run and the subband of that event. Thresholds for 10 noise FP per beam·hour: 6.62 (subbands) and 6.48 (full band).
Usage (pirate environment): export CHARTS_SUBBAND_RESULTS=FOLDER; python paired_comparison.py
FOLDER holds the two runs: completa_subbandas/ and completa_banda_completa/ (folder names of the existing runs).
Output: FOLDER/paired_comparison.md
"""
import csv
import os
import sys
import re
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if "CHARTS_SUBBAND_RESULTS" not in os.environ:
    sys.exit("ERROR: export CHARTS_SUBBAND_RESULTS=/folder/with/completa_subbandas/and/completa_banda_completa")
BASE = Path(os.environ["CHARTS_SUBBAND_RESULTS"])   # folder with completa_subbandas/ and completa_banda_completa/
DD = Path(__file__).resolve().parents[2] / "06_configs" / "charts_64b_dm1000.yml"
SEQ, DM_MIN, TOL_DM, TOL_T = 200, 20.0, 0.10, 0.05
RUNS = {"sb": ("completa_subbandas", 6.62), "fb": ("completa_banda_completa", 6.48)}


def delay_edge_to_top():
    import pirate_frb as pf
    cfg = pf.DedispersionConfig.from_yaml(str(DD))
    f_lo, f_hi = cfg.zone_freq_edges[0], cfg.zone_freq_edges[-1]
    return 1e-3 / cfg.dm_per_unit_delay() * (f_lo ** -2 - 501.6 ** -2) / (f_lo ** -2 - f_hi ** -2)


K_TOP = delay_edge_to_top()
ev_re = re.compile(r"beam_id=(\d+), dm=([\d.e+-]+), fpga_timestamp=(-?\d+), width=([\d.e+-]+) ms, "
                   r"subband=\[([\d.]+),([\d.]+)\] MHz, snr=([\d.e+-]+)")


def read_events(name):
    evs = []
    for m in map(ev_re.search, (BASE / name / "sifter.log").read_text(errors="replace").splitlines()):
        if m:
            evs.append((int(m[1]), float(m[2]), int(m[3]) / SEQ * 1e-3, float(m[4]), f"{m[5]}–{m[6]}", float(m[7])))
    b, dm, t, w, sb, s = map(np.array, zip(*evs))
    sel = dm >= DM_MIN
    return dict(b=b[sel].astype(int), dm=dm[sel], t=t[sel] - dm[sel] * K_TOP, w=w[sel], sb=sb[sel], s=s[sel])


def f(r, k, d=np.nan):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return d


EV = {k: read_events(n) for k, (n, _) in RUNS.items()}
run_sb = BASE / RUNS["sb"][0]
assign = run_sb / "assignment.csv" if (run_sb / "assignment.csv").exists() else run_sb / "asignacion.csv"
rows = [r for r in csv.DictReader(open(assign)) if float(r["dm"]) <= 1000]
res = []
for r in rows:
    b, dm = int(r["beam"]), float(r["dm"])
    t_abs = int(r.get("start_sample") or r["inicio_muestra"]) / 1000 + float(r["t_top_s"])
    box = (f(r, "boxcar_opt_ms", 0) or f(r, "boxcar_opt_visible_ms", 0)) / 1000
    d = dict(r=r, box_ms=box * 1e3)
    for k, e in EV.items():
        ok = (e["b"] == b) & (np.abs(e["dm"] - dm) <= max(TOL_DM * dm, 1.0)) & (np.abs(e["t"] - t_abs) <= max(TOL_T, box))
        if ok.any():
            i = np.flatnonzero(ok)[np.argmax(e["s"][ok])]
            d[k] = (e["s"][i], e["sb"][i], e["w"][i])
        else:
            d[k] = (0.0, "", 0.0)
    d["det_sb"], d["det_fb"] = d["sb"][0] >= RUNS["sb"][1], d["fb"][0] >= RUNS["fb"][1]
    res.append(d)

txt = "# T3e · Paired comparison, at 10 noise FP per beam·hour (6.62 subbands, 6.48 full band)\n\n"
txt += "| Set | n | detected by both | subbands only | full band only | neither |\n|---|---|---|---|---|---|\n"
for s in ("A", "B", "P", "NEG"):
    p = [d for d in res if d["r"]["set"] == s]
    c = Counter((d["det_sb"], d["det_fb"]) for d in p)
    txt += f"| {s} | {len(p)} | {c[(True, True)]} | {c[(True, False)]} | {c[(False, True)]} | {c[(False, False)]} |\n"

bands = sorted({d['sb'][1] for d in res if d['det_sb']})
txt += "\n## Subband of the best event, among those detected with subbands\n\n| Set | " + " | ".join(bands) + " |\n"
txt += "|---|" + "---|" * len(bands) + "\n"
for s in ("A", "B", "P", "NEG"):
    c = Counter(d["sb"][1] for d in res if d["r"]["set"] == s and d["det_sb"])
    txt += f"| {s} | " + " | ".join(str(c[x]) for x in bands) + " |\n"
txt += "\nRecovered (subbands only), by subband: " + str(dict(Counter(
    d["sb"][1] for d in res if d["det_sb"] and not d["det_fb"] and d["r"]["set"] != "NEG"))) + "\n"

# noise events: subband of every event ≥ threshold in the subband run
e = EV["sb"]
txt += "\nAll events ≥ 6.62 of the subband run (almost all noise or FRB), by subband: " + str(
    dict(Counter(e["sb"][e["s"] >= RUNS["sb"][1]]))) + "\n"

txt += "\n## Files recovered or lost with subbands (A, B, cut)\n\n"
txt += ("| file | set | exp. S/N | DM | boxcar ms | scat600 ms | emission channels | frac vis | S/N sb (subband) | S/N fb |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n")
for d in sorted([d for d in res if d["det_sb"] != d["det_fb"] and d["r"]["set"] != "NEG"],
                key=lambda d: (d["det_fb"], d["r"]["set"], -d["box_ms"])):
    r = d["r"]
    exp = f(r, "snr_visible_esperado") if r["set"] == "P" else f(r, "snr_esperado")
    txt += (f"| {r['name']} | {r['set']}{' (lost)' if d['det_fb'] else ''} | {exp:.1f} | {float(r['dm']):.0f} | "
            f"{d['box_ms']:.0f} | {f(r, 'scat_600_s') * 1e3:.1f} | {r['n_canales_emision']} | {r['frac_energia_visible']} | "
            f"{d['sb'][0]:.2f} ({d['sb'][1]}) | {d['fb'][0]:.2f} |\n")

# summary by emission bandwidth (A + B)
txt += "\n## A + B by emission channels (of 21 504)\n\n| channels | n | recall subbands | recall full band |\n|---|---|---|---|\n"
ab = [d for d in res if d["r"]["set"] in ("A", "B")]
for lo, hi in ((0, 5000), (5000, 10000), (10000, 15000), (15000, 21505)):
    p = [d for d in ab if lo <= f(d["r"], "n_canales_emision", 21504) < hi]
    if p:
        txt += f"| {lo}–{hi} | {len(p)} | {100 * np.mean([d['det_sb'] for d in p]):.1f} % | {100 * np.mean([d['det_fb'] for d in p]):.1f} % |\n"
(BASE / "paired_comparison.md").write_text(txt)
print(txt)
